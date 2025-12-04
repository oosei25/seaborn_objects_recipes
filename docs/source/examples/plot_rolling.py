"""
Rolling smoothing for multi-series
==================================
"""
import numpy as np
import pandas as pd
import seaborn.objects as so
import seaborn_objects_recipes as sor

rng = np.random.default_rng(0)
t = np.arange(200)
df = pd.DataFrame({
    "t": np.tile(t, 3),
    "y": np.r_[np.sin(t/15)+rng.normal(0,0.2,200),
               np.cos(t/18)+rng.normal(0,0.2,200),
               0.5*np.sin(t/20)+rng.normal(0,0.2,200)],
    "series": np.repeat(["A","B","C"], 200)
})

roll = sor.Rolling(window_type="gaussian", window_kwargs={"std": 3})

(so.Plot(df, x="t", y="y", color="series")
 .add(so.Lines(), so.Agg(), roll)
 .label(title="Rolling smoothing (gaussian, std=3)", x="t", y="y"))
