from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from seaborn._stats.base import Stat


@dataclass
class Lowess(Stat):
    """
    Perform locally-weighted regression (LOWESS) to smooth data.
    This statistical method allows fitting a smooth curve to your data
    using a local regression. It can be useful to visualize the trend of the data.

    Note that while this is based on the statsmodels implementation of lowess,
    the default value of `it` is set to 0. For a more outlier robust
    model increase the value of `it`.

    Parameters
    ----------
    frac : float, 0.2
        The fraction of data used when estimating each y-value.
    gridsize : int, 100
        The number of points in the grid to which the LOWESS is applied.
        Higher values result in a smoother curve.
    delta : float, 0.0
        Distance within which to use linear-interpolation instead of weighted regression.
    it : int, 0
        The number of iterations to perform. 0 = plain least-squares
        LOWESS; higher values re-weight outliers via a bisquare function at
        extra computational cost.
    num_bootstrap : int, optional
        The number of bootstrap samples to use for confidence intervals.
    alpha : float, 0.05
        Confidence level for the intervals.

    Returns
    -------
    DataFrame
        Columns “x”, “y” (smoothed) and, if bootstrapped, “ymin”/“ymax”.
    """

    frac: float = 0.2
    gridsize: int = 100
    delta: float = 0.0
    it: int = 0
    num_bootstrap: Optional[int] = None
    alpha: float = 0.05

    def __post_init__(self):
        # Type checking for the arguments
        if not isinstance(self.frac, float) or not (0 < self.frac <= 1):
            raise ValueError("frac must be a float between 0 and 1.")
        if not isinstance(self.gridsize, int) or self.gridsize <= 0:
            raise ValueError("gridsize must be a positive integer.")
        if self.num_bootstrap is not None and (
            not isinstance(self.num_bootstrap, int) or self.num_bootstrap <= 0
        ):
            raise ValueError("num_bootstrap must be a positive integer or None.")
        if not isinstance(self.alpha, float) or not (0 < self.alpha < 1):
            raise ValueError("alpha must be a float between 0 and 1.")
        if not isinstance(self.it, int) or self.it < 0:
            raise ValueError("iterations must be a non-negative integer.")
        if not isinstance(self.delta, float) or self.delta < 0:
            raise ValueError("delta must be a non-negative float.")
        if self.num_bootstrap is None and not self.alpha == 0.05:
            self.num_bootstrap = 200

    def _fit_predict(self, data):
        x = data["x"]
        xx = np.linspace(x.min(), x.max(), self.gridsize)
        result = sm.nonparametric.lowess(
            endog=data["y"],
            exog=x,
            frac=self.frac,
            delta=self.delta,
            it=self.it,
            xvals=xx,
        )
        if result.ndim == 1:  # Handle single-dimensional return values
            yy = result
        else:
            yy = result[:, 1]  # Select the predicted y-values
        return pd.DataFrame(dict(x=xx, y=yy))

    def _bootstrap_resampling(self, data) -> pd.DataFrame:
        xx = np.linspace(data["x"].min(), data["x"].max(), self.gridsize)
        bootstrap_estimates = np.empty((self.num_bootstrap, len(xx)))

        for i in range(self.num_bootstrap):
            sample = data.sample(frac=1, replace=True)
            result = sm.nonparametric.lowess(
                endog=sample["y"],
                exog=sample["x"],
                xvals=xx,
                frac=self.frac,
                delta=self.delta,
                it=self.it,
            )
            # Ensure the result is two-dimensional
            if result.ndim == 1:
                result = np.column_stack(
                    (xx, result)
                )  # Reformat to two-dimensional if needed
            bootstrap_estimates[i, :] = result[:, 1]

        lower_bound = np.percentile(bootstrap_estimates, self.alpha / 2 * 100, axis=0)
        upper_bound = np.percentile(
            bootstrap_estimates, (1 - (self.alpha / 2)) * 100, axis=0
        )

        return pd.DataFrame({"ymin": lower_bound, "ymax": upper_bound})

    def __call__(self, data: pd.DataFrame, groupby, orient, scales) -> pd.DataFrame:
        if orient == "x":
            xvar, yvar = data.columns[0], data.columns[1]
        else:
            xvar, yvar = data.columns[1], data.columns[0]

        df = data.rename(columns={xvar: "x", yvar: "y"}).dropna(subset=["x", "y"])

        unique_x = np.unique(df["x"])
        n = len(unique_x)

        k = 2
        min_frac = k / n
        if self.frac < min_frac:
            raise ValueError(
                f"`frac={self.frac:.3f}` is too small for only {n} distinct x‐values.\n"
                f"LOWESS needs at least ~{k+1} points per window, so try `frac` ≥ {min_frac:.3f}."
            )
        smoothed = self._fit_predict(df)

        grouping_vars = [str(v) for v in data if v in groupby.order]

        if not grouping_vars:
            # If no grouping variables, directly fit and predict
            smoothed = self._fit_predict(df)
        else:
            # Apply the fit_predict method for each group separately
            smoothed = groupby.apply(df, self._fit_predict)

        if self.num_bootstrap:
            if not grouping_vars:
                bootstrap_estimates = self._bootstrap_resampling(df)
            else:
                bootstrap_estimates = groupby.apply(df, self._bootstrap_resampling)

        return (
            smoothed.join(bootstrap_estimates[["ymin", "ymax"]])
            if self.num_bootstrap
            else smoothed
        )
