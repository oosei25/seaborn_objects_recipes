"""
LOWESS smoothing with optional bootstrap CIs
============================================

This example shows how to use :class:`seaborn_objects_recipes.Lowess` with and
without bootstrap confidence intervals.
"""

import seaborn as sns
import seaborn.objects as so
import seaborn_objects_recipes as sor

# Use a small CI bootstrap for faster docs builds
PENGUINS = sns.load_dataset("penguins").dropna(subset=["bill_length_mm", "body_mass_g"])

# --- With CI band --------------------------------------------------
low = sor.Lowess(frac=0.3, gridsize=150, num_bootstrap=200, alpha=0.05)
(
    so.Plot(PENGUINS, x="bill_length_mm", y="body_mass_g")
    .add(so.Dot(alpha=0.35))
    .add(so.Line(), low)
    .add(so.Band(), low)                 # uses ymin/ymax produced by the stat
    .label(
        title="LOWESS with 95% CI",
        x="Bill Length (mm)",
        y="Body Mass (g)",
    )
    .plot()
)

# --- Without CI band ------------------------------------------------
low2 = sor.Lowess(frac=0.3, gridsize=150)   # no bootstraps → no band
(
    so.Plot(PENGUINS, x="bill_length_mm", y="body_mass_g")
    .add(so.Dot(alpha=0.35))
    .add(so.Line(), low2)
    .label(
        title="LOWESS (no CI)",
        x="Bill Length (mm)",
        y="Body Mass (g)",
    )
    .plot()
)
