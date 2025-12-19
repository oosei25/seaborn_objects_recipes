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

## Features

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

## Recipes overview

`seaborn_objects_recipes` currently provides a small set of stats, moves, and marks
that plug directly into the `seaborn.objects` API.

### LOWESS smoother

 `seaborn_objects_recipes.recipes.lowess.Lowess`

A `Stat` that performs locally weighted regression using `statsmodels` and returns
a tidy DataFrame with `x`, `y` (smoothed) and, optionally, `ymin` / `ymax` columns
when interval estimation is enabled.

- `frac` controls how much of the data each local fit sees.
- `gridsize` controls how finely the curve is evaluated.
- `it` and `delta` mirror the underlying `statsmodels` parameters.
- If `num_bootstrap` is set (or implied), bootstrap resampling is used to form
  intervals, and `alpha` controls the tail probability used to define
  the bounds (smaller `alpha` → wider intervals).

  > The class also enforces a minimum feasible `frac` based on the number of distinct
  `x` values to avoid underdetermined fits.

### Polynomial fit with intervals

 `seaborn_objects_recipes.recipes.plotting.PolyFitWithCI`

A `Stat` that fits a polynomial of a given `order` and evaluates it on a regular
grid, returning `x`, `y`, `ymin`, and `ymax`.

- Uses normal-theory intervals based on the covariance of the fitted coefficients.
- `order` controls the polynomial degree, `gridsize` the resolution of the curve.
- `alpha` again controls the tail probability used for the band.

  > Designed to pair naturally with `so.Line()` and `so.Band()` for “fit + band”
  style visualizations.

### Rolling smoother

`seaborn_objects_recipes.recipes.rolling.Rolling`

A `Move` that applies a pandas rolling window operation along one axis of the plot.

- `window` sets the window size, `window_type` can select kernels
  (e.g. Gaussian), and `window_kwargs` (like `{"std": 2}`) configure them.
- `agg` chooses the aggregation (e.g. `"mean"`).

  > This is useful for smoothing trajectories or time series while keeping the
  original `seaborn.objects` structure (e.g. `Agg` + `Lines`) intact.

### Direct line labels

`seaborn_objects_recipes.recipes.line_label.LineLabel`

A `Mark` that places text directly on lines, so you can identify series without
relying only on a legend.

- Takes `text`, `color`, `alpha`, `fontsize`, `offset`, and
  `additional_distance_offset` to control label appearance and spacing.
- Internally works in screen coordinates and solves a small constrained
  optimization problem to separate labels, then maps them back into data space.

  > Intended for multi-line plots where direct labeling improves readability.

### Reference lines

`seaborn_objects_recipes.recipes.straight_line.StraightLine`

A `Mark` that draws simple horizontal or vertical reference lines using
`Axes.axline`.

- With `orient="x"`, it draws a vertical line; with `orient="y"`, a horizontal line.
- Inherits color and style from the usual `Mappable` properties
  (`color`, `linewidth`, `linestyle`, `alpha`).

  > Useful for adding thresholds, baselines, or other reference levels directly
  within the `seaborn.objects` grammar.

## Usage Notes & Correctness Guarantees

The smoothing utilities in this package extend Seaborn with production-friendly defaults for local regression and bootstrap-based uncertainty estimates. When bootstrapping is enabled (num_bootstrap > 0), the estimator returns confidence bounds in columns named exactly ymin and ymax, which integrate directly with the Confidence Band layer via:

```python

   (
      so.Plot(df, x="year", y="outcome")
      .add(so.Line(), low := sor.Lowess(frac=0.3, alpha=0.05) ) # Defaults to 200 bootstraps when alpha changes
      .add(so.Band(), low)  # Renders ymin/ymax if available
      .label(title="LOWESS with uncertainty", x="Year", y="Outcome")
   )
```

## When confidence intervals won’t show

- The input data has too few distinct x-values for the chosen frac.
- The smoothing window (frac) is below the minimum required coverage for stable local regression (frac ≥ k/n, where LOWESS needs ~3 points per window).
- The estimator runs but bootstrap bounds are skipped upstream due to `num_bootstrap` being None and not auto-defaulted when alpha changes.
- Numerical instability causes NaNs during kernel weighting (e.g., extreme delta values or all-zero weights).

## Package safety defaults

To ensure bootstrap bounds are generated when users adjust alpha without explicitly setting `num_bootstrap`, the estimator applies a guarded default:

```python
   # If user changed alpha but didn’t set num_bootstrap, assign a safe default
   if self.num_bootstrap is None and not self.alpha == 0.05:
      self.num_bootstrap = 200
```

> This preserves accurate CI generation while avoiding silent failures for common configuration mismatches. If your workflow needs tighter skepticism, consider increasing `num_bootstrap` (e.g., 1000–5000) when dataset size allows, or validating unique(x) counts before smoothing for faster feedback loops.

## Test coverage expectations

Core estimators are tested with:

- Synthetic sinusoidal data (plot_lowess_ci_gen)
- Real datasets via statsmodels LOWESS
- Boundary-condition tests for minimal CI coverage and frac constraints
- Faceting and orientation permutations for pooled/grouped frames
