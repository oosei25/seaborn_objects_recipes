from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd
from seaborn._stats.base import Stat


@dataclass
class PolyFitWithCI(Stat):
    """
    Fit a polynomial of the given order and resample data onto predicted curve
    including confidence intervals.

    Parameters
    ----------
    alpha : float
        The confidence level for the intervals.
    order : int
        The order of the polynomial to fit. Higher orders can capture more complex relationships.
    gridsize : int
        The number of points in the grid to which the polynomial is applied. Higher values result in a smoother curve.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame with the predicted curve's 'x', 'y', 'ymin', and 'ymax' coordinates.

    """

    alpha: float = 0.05
    order: int = 2
    gridsize: int = 100

    def __post_init__(self):
        # Type checking for the arguments
        if not isinstance(self.order, int) or self.order <= 0:
            raise ValueError("order must be a positive integer.")
        if not isinstance(self.gridsize, int) or self.gridsize <= 0:
            raise ValueError("gridsize must be a positive integer.")
        if not isinstance(self.alpha, float) or not (0 < self.alpha < 1):
            raise ValueError("alpha must be a float between 0 and 1.")

    def _fit_predict(self, data):
        x = data["x"].values
        y = data["y"].values
        if x.size <= self.order:
            xx = yy = []
        else:
            # Fit polynomial and create gridded values
            # for plotting.
            p = np.polyfit(x, y, self.order)
            xx = np.linspace(x.min(), x.max(), self.gridsize)
            yy = np.polyval(p, xx)

            # Calculate confidence intervals

            # Design matrix for fitting
            X_to_fit = np.vander(x, self.order + 1)

            # Calculate standard errors
            y_hat = np.polyval(p, x)
            residuals = y - y_hat
            dof = max(0, len(x) - (self.order + 1))
            residual_std_error = np.sqrt(np.sum(residuals**2) / dof)

            # Covariance matrix
            C_matrix = np.linalg.inv(X_to_fit.T @ X_to_fit) * residual_std_error**2

            # Design matrix for prediction points
            X_design = np.vander(xx, self.order + 1)

            # Calculate the standard error for the predicted values
            y_err = np.sqrt(np.sum((X_design @ C_matrix) * X_design, axis=1))

            # Calculate the confidence intervals using NormalDist
            z_score = NormalDist().inv_cdf(1 - self.alpha / 2)
            ci = y_err * z_score
            ci_lower = yy - ci
            ci_upper = yy + ci

        results = pd.DataFrame(dict(x=xx, y=yy, ymin=ci_lower, ymax=ci_upper))

        return results

    def __call__(self, data, groupby, orient, scales):
        # Rename columns to match expected input for _fit_predict
        if orient == "x":
            xvar = data.columns[0]
            yvar = data.columns[1]
        else:
            xvar = data.columns[1]
            yvar = data.columns[0]

        renamed_data = data.rename(columns={xvar: "x", yvar: "y"})
        return groupby.apply(renamed_data.dropna(subset=["x", "y"]), self._fit_predict)
