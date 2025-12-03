"""Tests for PolyFitWithCI confidence interval calculations."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from seaborn._core.groupby import GroupBy

from seaborn_objects_recipes.recipes.plotting import PolyFitWithCI


@pytest.fixture
def quadratic_data():
    """Generate data with a known quadratic relationship and normal noise."""
    np.random.seed(42)
    n_points = 10_000
    x = np.linspace(0, 10, n_points)
    # Quadratic: y = 0.5*x^2 - 2*x + 5
    a, b, c = 0.5, -2.0, 5.0
    y_true = a * x**2 + b * x + c
    noise_std = 2.0
    y = y_true + np.random.normal(0, noise_std, n_points)
    return pd.DataFrame({"x": x, "y": y}), (a, b, c), noise_std


def _eval_stat(stat, df, x, y, orient="x"):
    """Run a seaborn.objects Stat on an (x, y) dataframe with no real grouping."""
    df_xy = df.rename(columns={x: "x", y: "y"}).dropna(subset=["x", "y"])
    fake_groupby = GroupBy(["__dummy__"])  # not present in df_xy -> no grouping
    return stat(df_xy, fake_groupby, orient, {})


class TestPolyFitWithCI:
    """Test suite for PolyFitWithCI confidence interval calculations."""

    @pytest.mark.parametrize("alpha", [0.10, 0.05, 0.01])
    def test_compare_with_statsmodels(self, quadratic_data, alpha):
        """
        Compare PolyFitWithCI confidence intervals with statsmodels OLS.

        Tests that our polynomial fit with CIs produces similar results to
        statsmodels' OLS regression with prediction intervals.

        Tests multiple confidence levels: 90%, 95%, and 99%.
        """
        df, (a, b, c), noise_std = quadratic_data

        # Fit with PolyFitWithCI
        polyfit = PolyFitWithCI(order=2, gridsize=50, alpha=alpha)
        result = _eval_stat(polyfit, df, "x", "y")

        # Fit with statsmodels OLS
        X = np.column_stack([df["x"] ** 2, df["x"], np.ones(len(df))])
        model = sm.OLS(df["y"], X).fit()

        # Get predictions at the same grid points
        X_pred = np.column_stack([result["x"] ** 2, result["x"], np.ones(len(result))])
        predictions = model.get_prediction(X_pred)
        pred_summary = predictions.summary_frame(alpha=alpha)

        # Compare fitted values
        sm_fitted = pred_summary["mean"].values
        our_fitted = result["y"].values

        # Fitted values should be very close (independent of alpha)
        np.testing.assert_allclose(
            our_fitted,
            sm_fitted,
            rtol=1e-10,
            err_msg=f"For alpha={alpha}: Fitted values differ from statsmodels",
        )

        # Compare confidence intervals
        sm_lower = pred_summary["mean_ci_lower"].values
        sm_upper = pred_summary["mean_ci_upper"].values
        our_lower = result["ymin"].values
        our_upper = result["ymax"].values

        # CI bounds should be close (allow small numerical differences)
        np.testing.assert_allclose(
            our_lower,
            sm_lower,
            atol=1e-4,
            err_msg=f"For alpha={alpha}: Lower CI bounds differ significantly from statsmodels",
        )
        np.testing.assert_allclose(
            our_upper,
            sm_upper,
            atol=1e-4,
            err_msg=f"For alpha={alpha}: Upper CI bounds differ significantly from statsmodels",
        )

    def test_ci_validity(self, quadratic_data):
        """Test that confidence intervals are valid (ymin <= y <= ymax)."""
        df, _, _ = quadratic_data

        polyfit = PolyFitWithCI(order=2, gridsize=50, alpha=0.05)
        result = _eval_stat(polyfit, df, "x", "y")

        # Check structure
        assert "ymin" in result.columns
        assert "ymax" in result.columns
        assert len(result) == 50

        # CIs should be valid
        assert (result["ymax"] >= result["ymin"]).all()
        assert (
            result["ymax"] > result["ymin"]
        ).mean() > 0.95  # Most should be proper intervals

    @pytest.mark.parametrize(
        "alpha,expected_coverage",
        [
            (0.10, 0.90),  # 90% CI
            (0.05, 0.95),  # 95% CI
            (0.01, 0.99),  # 99% CI
        ],
    )
    def test_alpha_affects_width(self, quadratic_data, alpha, expected_coverage):
        """Test that different alpha values produce appropriately sized intervals."""
        df, (a, b, c), noise_std = quadratic_data

        # Fit with given alpha
        polyfit = PolyFitWithCI(order=2, gridsize=50, alpha=alpha)
        result = _eval_stat(polyfit, df, "x", "y")

        # Calculate true function values at grid points
        x_grid = result["x"].values
        y_true = a * x_grid**2 + b * x_grid + c

        # Check coverage
        within_ci = (y_true >= result["ymin"]) & (y_true <= result["ymax"])
        coverage = within_ci.mean()

        # Allow ±0.15 tolerance (more generous than LOWESS due to polynomial fitting)
        assert abs(coverage - expected_coverage) < 0.15, (
            f"For alpha={alpha}, coverage is {coverage:.1%}, "
            f"expected approximately {expected_coverage:.1%}"
        )

    def test_ci_width_comparison_across_alphas(self, quadratic_data):
        """Test that CI width decreases as confidence level decreases."""
        df, _, _ = quadratic_data

        # Fit with different alphas
        polyfit_90 = PolyFitWithCI(order=2, gridsize=50, alpha=0.10)
        polyfit_95 = PolyFitWithCI(order=2, gridsize=50, alpha=0.05)
        polyfit_99 = PolyFitWithCI(order=2, gridsize=50, alpha=0.01)

        result_90 = _eval_stat(polyfit_90, df, "x", "y")
        result_95 = _eval_stat(polyfit_95, df, "x", "y")
        result_99 = _eval_stat(polyfit_99, df, "x", "y")

        width_90 = (result_90["ymax"] - result_90["ymin"]).mean()
        width_95 = (result_95["ymax"] - result_95["ymin"]).mean()
        width_99 = (result_99["ymax"] - result_99["ymin"]).mean()

        # 90% CI should be narrower than 95% CI, which should be narrower than 99% CI
        assert width_90 < width_95 < width_99, (
            f"CI widths not in expected order: "
            f"90%={width_90:.2f}, 95%={width_95:.2f}, 99%={width_99:.2f}"
        )

    def test_edge_effects(self, quadratic_data):
        """Test that confidence intervals are wider at the edges (typical for polynomial fits)."""
        df, _, _ = quadratic_data

        polyfit = PolyFitWithCI(order=2, gridsize=50, alpha=0.05)
        result = _eval_stat(polyfit, df, "x", "y")

        ci_width = result["ymax"] - result["ymin"]

        # Width at edges (first and last 5 points)
        edge_width = pd.concat([ci_width.iloc[:5], ci_width.iloc[-5:]]).mean()

        # Width at center
        center_width = ci_width.iloc[20:30].mean()

        # Edge should typically be wider than center for polynomial regression
        assert (
            edge_width > center_width * 0.8
        ), "Edge CIs should not be significantly narrower than center CIs"

    def test_high_order_polynomial(self):
        """Test that higher order polynomials work correctly."""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        # Cubic: y = 0.1*x^3 - 0.5*x^2 + 2*x + 1
        y = 0.1 * x**3 - 0.5 * x**2 + 2 * x + 1 + np.random.normal(0, 1, 100)
        df = pd.DataFrame({"x": x, "y": y})

        # Fit with order=3
        polyfit = PolyFitWithCI(order=3, gridsize=50, alpha=0.05)
        result = _eval_stat(polyfit, df, "x", "y")

        # Should produce valid results
        assert len(result) == 50
        assert "ymin" in result.columns
        assert "ymax" in result.columns
        assert (result["ymax"] >= result["ymin"]).all()
