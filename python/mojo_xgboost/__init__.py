"""Dense histogram gradient boosting powered by Mojo."""

from .core import Booster, DMatrix
from .sklearn import XGBClassifier, XGBModel, XGBRegressor
from .training import train

__all__ = [
    "Booster",
    "DMatrix",
    "XGBClassifier",
    "XGBModel",
    "XGBRegressor",
    "train",
]

__version__ = "0.1.0"
