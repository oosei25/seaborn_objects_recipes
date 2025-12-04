# Seaborn Objects Recipes

```{image} https://img.shields.io/pypi/v/seaborn_objects_recipes.svg
:target: https://pypi.org/project/seaborn_objects_recipes/
```

```{image} https://img.shields.io/pypi/pyversions/seaborn_objects_recipes.svg
```

```{image} https://github.com/oosei25/seaborn_objects_recipes/actions/workflows/actions.yml/badge.svg
:target: https://github.com/oosei25/seaborn_objects_recipes/actions
```

```{image} https://img.shields.io/badge/docs-stable-blue
:target: https://oosei25.github.io/seaborn_objects_recipes/
```

## Features:

- LOWESS with optional bootstrapped CIs
- Polynomial fits with bands
- Rolling smoothing
- Line labeling & straight lines

```{toctree}
:maxdepth: 1
:caption: Gallery

auto_examples/index

:maxdepth: 1
:caption: API

api/index
```

## Usage Notes & Correctness Guarantees

The smoothing utilities in this package extend Seaborn with production-friendly defaults for local regression and bootstrap-based uncertainty estimates. When bootstrapping is enabled (num_bootstrap > 0), the estimator returns confidence bounds in columns named exactly ymin and ymax, which integrate directly with the Confidence Band layer via:

```python
   low = sor.Lowess(frac=0.3, alpha=0.05)  # Defaults to 200 bootstraps when alpha changes
   (
      so.Plot(df, x="year", y="outcome")
      .add(so.Line(), low)
      .add(so.Band(), low)  # Renders ymin/ymax if available
      .label(title="LOWESS with uncertainty", x="Year", y="Outcome")
   )
```

> When confidence intervals won’t show

• The input data has too few distinct x-values for the chosen frac.

• The smoothing window (frac) is below the minimum required coverage for stable local regression (frac ≥ k/n, where LOWESS needs ~3 points per window).

• The estimator runs but bootstrap bounds are skipped upstream due to num_bootstrap being None and not auto-defaulted when alpha changes.

• Numerical instability causes NaNs during kernel weighting (e.g., extreme delta values or all-zero weights).


## Package safety defaults

To ensure bootstrap bounds are generated when users adjust alpha without explicitly setting num_bootstrap, the estimator applies a guarded default:

```python
   # If user changed alpha but didn’t set num_bootstrap, assign a safe default
   if self.num_bootstrap is None and not self.alpha == 0.05:
      self.num_bootstrap = 200
```

>> This preserves accurate CI generation while avoiding silent failures for common configuration mismatches. If your workflow needs tighter skepticism, consider increasing num_bootstrap (e.g., 1000–5000) when dataset size allows, or validating unique(x) counts before smoothing for faster feedback loops.

## Test coverage expectations

Core estimators are tested with:

• Synthetic sinusoidal data (plot_lowess_ci_gen)

• Real datasets via statsmodels LOWESS

• Boundary-condition tests for minimal CI coverage and frac constraints

• Faceting and orientation permutations for pooled/grouped frames
