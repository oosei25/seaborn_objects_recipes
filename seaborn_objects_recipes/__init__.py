"""The recipes module provides a collection of recipes that can be used
together with the seaborn library to create custom plots."""

# try:
#     import seaborn as sns
#     import statsmodels.api as sm
#     import scipy as sp
#     import seaborn.objects as so
# except ImportError:
#     raise ImportError(
#         "The recipes module requires seaborn [>= 0.12.0], statsmodels, scipy"
#     )

from .recipes.rolling import Rolling  # noqa: F401

from .recipes.line_label import LineLabel  # noqa: F401

from .recipes.lowess import Lowess  # noqa: F401

from .recipes.plotting import PolyFitWithCI # noqa: F401

__all__ = ['Rolling', 'LineLabel', 'Lowess', 'PolyFitWithCI']

