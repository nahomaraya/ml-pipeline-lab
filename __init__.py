# Expose a simple library-like API

from .data_preprocessing import preprocess_data
from .linear_regression import train_linear_regression
from .logistic_regression_model import train_logistic_regression
from .decision_tree_model import train_decision_tree
from .random_forest_model import train_random_forest

__all__ = [
    "preprocess_data",
    "train_linear_regression",
    "train_logistic_regression",
    "train_decision_tree",
    "train_random_forest",
]


