"""
Predspot — predicting crime hotspots with machine learning.

Typical use::

    from predspot import Dataset, PredictionPipeline
    from predspot.crime_mapping import KDE, create_gridpoints
    from predspot.feature_engineering import Seasonality, Trend, Diff
    from predspot.utilities import PandasFeatureUnion
"""

from predspot import (
    crime_mapping,
    dataset_preparation,
    feature_engineering,
    ml_modelling,
    synthetic,
    utilities,
)
from predspot.crime_mapping import (
    KDE,
    QuadratCount,
    create_gridhexagonal,
    create_gridpoints,
    create_gridsquares,
    get_city_shape,
    load_study_area,
)
from predspot.dataset_preparation import Dataset
from predspot.ml_modelling import PredictionPipeline
from predspot.synthetic import generate_crimes
from predspot.utilities import PandasFeatureUnion

__version__ = "0.2.0"

__all__ = [
    "Dataset",
    "PredictionPipeline",
    "PandasFeatureUnion",
    "KDE",
    "QuadratCount",
    "create_gridpoints",
    "create_gridhexagonal",
    "create_gridsquares",
    "load_study_area",
    "get_city_shape",
    "generate_crimes",
    "synthetic",
    "crime_mapping",
    "dataset_preparation",
    "feature_engineering",
    "ml_modelling",
    "utilities",
]
