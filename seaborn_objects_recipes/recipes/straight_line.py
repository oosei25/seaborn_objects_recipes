from dataclasses import dataclass

import matplotlib as mpl
from seaborn._marks.base import (
    Mappable,
    MappableColor,
    MappableFloat,
    MappableString,
    Mark,
    resolve_color,
    resolve_properties,
)


@dataclass
class StraightLine(Mark):
    """Object drawing an horizontal or vertical line using the axline.
    Giving orient "x" will result in a vertical line.
    """

    color: MappableColor = Mappable("C0")
    alpha: MappableFloat = Mappable(1)
    linewidth: MappableFloat = Mappable(rc="lines.linewidth")
    linestyle: MappableString = Mappable(rc="lines.linestyle")

    def _plot(self, split_gen, scales, orient):

        for keys, data, ax in split_gen():

            vals = resolve_properties(self, keys, scales)
            vals["color"] = resolve_color(self, keys, scales=scales)

            artist_kws = self.artist_kws.copy()
            value = {"x": "y", "y": "x"}[orient]
            xy1_dict = {value: float(data[value].to_numpy()), orient: 0}
            xy2_dict = {value: float(data[value].to_numpy()), orient: 1}
            ax.axline(
                (xy1_dict["x"], xy1_dict["y"]),
                (xy2_dict["x"], xy2_dict["y"]),
                color=vals["color"],
                linewidth=vals["linewidth"],
                linestyle=vals["linestyle"],
                **artist_kws,
            )

    def _legend_artist(self, variables, value, scales):

        keys = {v: value for v in variables}
        vals = resolve_properties(self, keys, scales)
        vals["color"] = resolve_color(self, keys, scales=scales)

        artist_kws = self.artist_kws.copy()

        return mpl.lines.Line2D(
            [],
            [],
            color=vals["color"],
            linewidth=vals["linewidth"],
            linestyle=vals["linestyle"],
            **artist_kws,
        )
