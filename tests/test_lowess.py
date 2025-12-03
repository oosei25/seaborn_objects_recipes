"""Tests for Lowess confidence interval calculations comparing with statsmodels."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from seaborn._core.groupby import GroupBy

from seaborn_objects_recipes.recipes.lowess import Lowess


@pytest.fixture
def sinusoidal_data():
    """Generate data with a sinusoidal relationship and normal noise."""
    np.random.seed(42)
    n_points = 10_000
    x = np.linspace(0, 4 * np.pi, n_points)
    # Sinusoidal with linear trend: y = sin(x) + 0.3*x
    y_true = np.sin(x) + 0.3 * x
    noise_std = 0.5
    y = y_true + np.random.normal(0, noise_std, n_points)
    return pd.DataFrame({"x": x, "y": y}), noise_std


@pytest.fixture
def exponential_data():
    """Generate data with an exponential relationship and normal noise."""
    np.random.seed(123)
    n_points = 10_000
    x = np.linspace(0, 5, n_points)
    # Exponential: y = 2 * exp(0.3*x)
    y_true = 2 * np.exp(0.3 * x)
    noise_std = 0.5
    y = y_true + np.random.normal(0, noise_std, n_points)
    return pd.DataFrame({"x": x, "y": y}), noise_std


def _eval_stat(stat, df, x, y, orient="x"):
    """Run a seaborn.objects Stat on an (x, y) dataframe with no real grouping."""
    df_xy = df.rename(columns={x: "x", y: "y"}).dropna(subset=["x", "y"])
    fake_groupby = GroupBy(["__dummy__"])  # not present in df_xy -> no grouping
    return stat(df_xy, fake_groupby, orient, {})


class TestLowessComparison:
    """Test suite comparing Lowess with statsmodels implementation."""

    def test_compare_smoothed_curve_with_statsmodels(self, sinusoidal_data):
        """
        Compare our Lowess smoothed curve with statsmodels lowess.

        The smoothed curves should be very similar since both use the same algorithm.
        """
        df, noise_std = sinusoidal_data

        # Fit with our Lowess (no bootstrap for this test)
        lowess = Lowess(frac=0.3, it=0, gridsize=100, num_bootstrap=None)
        result = _eval_stat(lowess, df, "x", "y")

        # Fit with statsmodels lowess at the same grid points
        sm_result = sm.nonparametric.lowess(
            endog=df["y"], exog=df["x"], frac=0.3, it=0, xvals=result["x"].values
        )

        # Extract smoothed values
        if sm_result.ndim == 1:
            sm_fitted = sm_result
        else:
            sm_fitted = sm_result[:, 1]

        our_fitted = result["y"].values

        # Smoothed curves should be very close
        np.testing.assert_allclose(
            our_fitted,
            sm_fitted,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Smoothed curves differ from statsmodels",
        )

    @pytest.mark.parametrize(
        "alpha",
        [
            0.20,
            0.10,
            0.05,
            0.01,
        ],
    )
    def test_bootstrap_ci_coverage(self, sinusoidal_data, alpha):
        """
        Test that bootstrap confidence intervals achieve appropriate coverage.

        For data with normal noise, the bootstrap CIs should contain the true
        smooth function at approximately (1-alpha)% of points.
        """
        df, noise_std = sinusoidal_data

        # Fit with our Lowess with bootstrap
        lowess = Lowess(frac=0.3, gridsize=100, num_bootstrap=1_000, alpha=alpha)
        result = _eval_stat(lowess, df, "x", "y")

        # Re-do bootstrap
        fits = []
        for _ in range(1_000):
            resampled = df.sample(frac=1.0, replace=True)
            statsmodels_run = sm.nonparametric.lowess(
                endog=resampled["y"],
                exog=resampled["x"],
                frac=0.3,
                xvals=result["x"].values,
                it=0,
            )

            # Check coverage
            within_ci = (statsmodels_run >= result["ymin"]) & (
                statsmodels_run <= result["ymax"]
            )
            fit_coverage = within_ci.mean()

            fits.append(fit_coverage)

        overall_coverage = np.mean(fits)

        # Check damn close for this many bootstraps
        assert abs(overall_coverage - (1 - alpha)) < 0.01, (
            f"For alpha={alpha}, coverage is {overall_coverage:.1%}, "
            f"expected approximately {(1 - alpha):.1%}"
        )

    def test_exponential_data(self, exponential_data):
        """Test Lowess on exponential data."""
        df, noise_std = exponential_data

        # Fit with our Lowess
        lowess = Lowess(frac=0.3, gridsize=100, num_bootstrap=None)
        result = _eval_stat(lowess, df, "x", "y")

        # Fit with statsmodels lowess
        sm_result = sm.nonparametric.lowess(
            endog=df["y"], exog=df["x"], frac=0.3, xvals=result["x"].values, it=0
        )

        if sm_result.ndim == 1:
            sm_fitted = sm_result
        else:
            sm_fitted = sm_result[:, 1]

        our_fitted = result["y"].values

        # Should match for exponential data too
        np.testing.assert_allclose(
            our_fitted,
            sm_fitted,
            rtol=1e-6,
            atol=1e-3,  # Slightly larger tolerance for exponential
            err_msg="Smoothed curves differ from statsmodels on exponential data",
        )

    def test_with_iterations(self, sinusoidal_data):
        """Test that Lowess works with robust iterations parameter."""
        df, noise_std = sinusoidal_data

        # Add some outliers
        df_with_outliers = df.copy()
        outlier_idx = np.random.choice(len(df), size=50, replace=False)
        df_with_outliers.loc[outlier_idx, "y"] += np.random.choice([-5, 5], size=50)

        # Fit with iterations (robust to outliers)
        lowess = Lowess(frac=0.3, gridsize=100, num_bootstrap=None, it=3)
        result = _eval_stat(lowess, df_with_outliers, "x", "y")

        # Fit with statsmodels lowess with iterations
        sm_result = sm.nonparametric.lowess(
            endog=df_with_outliers["y"],
            exog=df_with_outliers["x"],
            frac=0.3,
            it=3,
            xvals=result["x"].values,
        )

        if sm_result.ndim == 1:
            sm_fitted = sm_result
        else:
            sm_fitted = sm_result[:, 1]

        our_fitted = result["y"].values

        # Should match even with iterations
        np.testing.assert_allclose(
            our_fitted,
            sm_fitted,
            rtol=1e-5,
            atol=1e-3,
            err_msg="Smoothed curves differ from statsmodels with iterations",
        )

    def test_small_frac_large_data(self):
        """Test Lowess with small frac on large dataset."""
        np.random.seed(42)
        n_points = 1000
        x = np.linspace(0, 10, n_points)
        y = np.sin(x) + np.random.normal(0, 0.3, n_points)
        df = pd.DataFrame({"x": x, "y": y})

        # Small frac = less smoothing
        lowess = Lowess(frac=0.1, gridsize=50, num_bootstrap=None)
        result = _eval_stat(lowess, df, "x", "y")

        # Compare with statsmodels
        sm_result = sm.nonparametric.lowess(
            endog=df["y"], exog=df["x"], frac=0.1, xvals=result["x"].values, it=0
        )

        if sm_result.ndim == 1:
            sm_fitted = sm_result
        else:
            sm_fitted = sm_result[:, 1]

        np.testing.assert_allclose(
            result["y"].values,
            sm_fitted,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Small frac results differ from statsmodels",
        )
