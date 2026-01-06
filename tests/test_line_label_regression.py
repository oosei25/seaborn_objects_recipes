import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn.objects as so
import matplotlib.text as mtext

import seaborn_objects_recipes as sor


def test_line_label_handles_duplicate_max_position():
    """
    Regression test: LineLabel should pick a single row per group even when
    multiple rows share the group's max x (avoids passing array positions to
    matplotlib Text, which can crash during rendering).
    """

    # Construct data where each group has TWO rows at the max x (= 2)
    df = pd.DataFrame(
        {
            "x": [0, 1, 2, 2, 0, 1, 2, 2],
            "y": [0.0, 0.5, 1.0, 1.1, 0.2, 0.6, 0.9, 1.0],
            "series": ["sin"] * 4 + ["cos"] * 4,
        }
    )

    p = (
        so.Plot(df, x="x", y="y", color="series")
        .add(so.Line())
        .add(sor.LineLabel(text="series", offset=8))
    )

    # Trigger a full draw (this is where CI used to blow up)
    plotter = p.plot()

    fig = getattr(plotter, "figure", None) or getattr(plotter, "_figure", None)
    assert fig is not None, "Could not access matplotlib Figure from seaborn Plotter"

    fig.canvas.draw()

    # Sanity checks: we should have one label per series
    texts = [t for t in fig.findobj(mtext.Text) if t.get_text() in {"sin", "cos"}]
    assert len(texts) == 2

    # And their x/y positions must be scalar-like (not arrays)
    for t in texts:
        x, y = t.get_position()
        assert np.isscalar(x)
        assert np.isscalar(y)
