import dataclasses
import operator
from typing import Any, ClassVar

import pandas as pd
import seaborn.objects as so
from seaborn._core.groupby import GroupBy


@dataclasses.dataclass
class Rolling(so.Move):
    window: int = 2
    window_type: str | None = None
    window_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    agg: str = "mean"

    group_by_orient: ClassVar[bool] = False

    def _rolling(self, df: pd.DataFrame, var: str) -> pd.DataFrame:
        aggregate = operator.methodcaller(self.agg, **self.window_kwargs)

        df[var] = aggregate(
            df[var].rolling(
                window=self.window,
                min_periods=1,
                win_type=self.window_type,
                closed="neither",
            ),
        )
        return df

    def __call__(
        self,
        data: pd.DataFrame,
        groupby: GroupBy,
        orient: str,
        scales: dict[str, so.Scale],
    ) -> pd.DataFrame:
        del scales
        other = {"x": "y", "y": "x"}[orient]
        return groupby.apply(data, self._rolling, other)
