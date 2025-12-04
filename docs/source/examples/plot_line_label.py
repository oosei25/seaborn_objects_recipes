"""
Line labeling at series endpoints
=================================
"""

import numpy as np
import pandas as pd
import seaborn.objects as so
import seaborn_objects_recipes as sor

t = np.arange(0, 50)
df = pd.DataFrame({
    "t": np.tile(t, 3),
    "y": np.r_[t, t*0.6 + 5, t*1.2 - 7],
    "series": np.repeat(["A", "B", "C"], t.size),
})

roll = sor.Rolling(window_type="gaussian", window_kwargs={"std": 2})

(
    so.Plot(df, x="t", y="y", color="series", text="series")
    .add(so.Lines(), so.Agg(), roll)
    .add(sor.LineLabel(offset=6), so.Agg(), roll)
    .label(title="Rolling smoothed lines with endpoint labels", x="t", y="y")
    .plot()
)
