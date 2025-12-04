"""
Polynomial fit with confidence band
===================================
"""
import numpy as np, pandas as pd
import seaborn.objects as so
import seaborn_objects_recipes as sor

rng = np.random.default_rng(0)
x = np.linspace(0, 2*np.pi, 150)
y = np.sin(x) + rng.normal(0, 0.2, size=x.size)
df = pd.DataFrame({"x": x, "y": y})

pf = sor.PolyFitWithCI(order=2, gridsize=200, alpha=0.05)

(
    so.Plot(df, x="x", y="y")
    .add(so.Dot(alpha=0.35))
    .add(so.Line(), pf)
    .add(so.Band(), pf)
    .label(title="Quadratic fit ± 95% CI", x="x", y="y")
    .plot()
)
