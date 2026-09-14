"""
Pipeline Module
===============

Convenience functions to run a sensible default prediction pipeline in one
call, and to generate small synthetic datasets for quick experiments.

Example:
    >>> from predspot.pipeline import generate_testdata, run_prediction_pipeline
    >>> crimes, study_area = generate_testdata(2000, '2019-01-01', '2020-12-31', seed=0)
    >>> predictions, pipeline = run_prediction_pipeline(crimes, study_area, grid_resolution=1)
"""

__author__ = "Adelson Araujo"

import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from predspot import crime_mapping, dataset_preparation, feature_engineering, ml_modelling
from predspot.synthetic import generate_crimes
from predspot.utilities import PandasFeatureUnion

logger = logging.getLogger(__name__)

# A ~10 x 10 km box (west, south, east, north) used as default study area.
DEFAULT_BOUNDS = (-35.30, -5.90, -35.20, -5.80)


def generate_testdata(n_points, start_time, end_time, bounds=DEFAULT_BOUNDS, seed=None):
    """
    Generate synthetic crime events inside a rectangular study area.

    A thin wrapper around :func:`predspot.synthetic.generate_crimes` with
    three hotspots and the default temporal patterns.

    Args:
        n_points (int): Number of events.
        start_time (str): First possible timestamp (``'YYYY-MM-DD'``).
        end_time (str): Last possible timestamp (``'YYYY-MM-DD'``).
        bounds (tuple): ``(west, south, east, north)`` in WGS84 degrees.
        seed (int, optional): Seed for reproducibility.

    Returns:
        tuple: ``(crimes, study_area)`` — a DataFrame with ``tag``, ``t``,
        ``lon``, ``lat`` and a one-row GeoDataFrame with the study area.
    """
    west, south, east, north = bounds
    study_area = gpd.GeoDataFrame(
        {"name": ["study_area"]}, geometry=[box(west, south, east, north)], crs="EPSG:4326"
    )
    crimes = generate_crimes(
        study_area, n_events=n_points, start=start_time, end=end_time, seed=seed
    )
    return crimes, study_area


def build_default_pipeline(
    study_area, tfreq="M", grid_resolution=1, lags=2, bandwidth="silverman", random_state=None
):
    """
    Build the default Predspot pipeline: KDE mapping, seasonal/trend/diff
    features, quantile scaling, RFE feature selection and a random forest.

    Args:
        study_area (GeoDataFrame): Study area used to build the point grid.
        tfreq (str): Time frequency (``'M'``, ``'W'`` or ``'D'``).
        grid_resolution (float): Grid spacing in kilometers.
        lags (int): Number of lags (and STL period) of the features.
        bandwidth (str or float): KDE bandwidth, see :class:`predspot.crime_mapping.KDE`.
        random_state (int, optional): Seed for the estimator and shuffling.

    Returns:
        predspot.ml_modelling.PredictionPipeline: An unfitted pipeline.
    """
    grid = crime_mapping.create_gridpoints(study_area, grid_resolution)
    return ml_modelling.PredictionPipeline(
        mapping=crime_mapping.KDE(tfreq=tfreq, grid=grid, bandwidth=bandwidth),
        fextraction=PandasFeatureUnion(
            [
                ("seasonal", feature_engineering.Seasonality(lags=lags, tfreq=tfreq)),
                ("trend", feature_engineering.Trend(lags=lags, tfreq=tfreq)),
                ("diff", feature_engineering.Diff(lags=lags, tfreq=tfreq)),
            ]
        ),
        estimator=Pipeline(
            [
                (
                    "f_scaling",
                    feature_engineering.FeatureScaling(
                        QuantileTransformer(n_quantiles=10, output_distribution="uniform")
                    ),
                ),
                (
                    "f_selection",
                    ml_modelling.FeatureSelection(
                        RFE(RandomForestRegressor(n_estimators=20, random_state=random_state))
                    ),
                ),
                (
                    "model",
                    ml_modelling.Model(
                        RandomForestRegressor(n_estimators=50, random_state=random_state)
                    ),
                ),
            ]
        ),
        random_state=random_state,
    )


def run_prediction_pipeline(
    crime_data,
    study_area,
    crime_tags=None,
    time_range=None,
    tfreq="M",
    grid_resolution=1,
    lags=2,
    random_state=None,
):
    """
    Fit the default pipeline on crime data and forecast the next period.

    Args:
        crime_data (pandas.DataFrame): Events with ``tag``, ``t``, ``lon``, ``lat``.
        study_area (GeoDataFrame): Study area boundary.
        crime_tags (list, optional): Keep only these crime types.
        time_range (tuple, optional): ``('HH:MM', 'HH:MM')`` to keep only
            events within this time of day.
        tfreq (str): Time frequency (``'M'``, ``'W'`` or ``'D'``).
        grid_resolution (float): Grid spacing in kilometers.
        lags (int): Number of lags (and STL period) of the features.
        random_state (int, optional): Seed for reproducibility.

    Returns:
        tuple: ``(predictions, pipeline)`` — the forecast for the next period
        and the fitted :class:`predspot.ml_modelling.PredictionPipeline`.
    """
    missing = [c for c in ("tag", "t", "lat", "lon") if c not in crime_data.columns]
    if missing:
        raise ValueError(f"Crime data must contain columns tag, t, lat, lon; missing {missing}")
    if crime_tags:
        crime_data = crime_data.loc[crime_data["tag"].isin(crime_tags)]
    if time_range:
        time_ix = pd.DatetimeIndex(pd.to_datetime(crime_data["t"]))
        crime_data = crime_data.iloc[time_ix.indexer_between_time(time_range[0], time_range[1])]

    dataset = dataset_preparation.Dataset(crimes=crime_data, study_area=study_area)
    pipeline = build_default_pipeline(
        study_area,
        tfreq=tfreq,
        grid_resolution=grid_resolution,
        lags=lags,
        random_state=random_state,
    )
    pipeline.fit(dataset)
    predictions = pipeline.predict()
    return predictions, pipeline


def evaluate_pipeline(pipeline, scoring="r2", cv=5):
    """
    Cross-validate a fitted pipeline; see :meth:`PredictionPipeline.evaluate`.

    Args:
        pipeline (PredictionPipeline): A fitted pipeline.
        scoring (str): ``'r2'`` or ``'mse'``.
        cv (int): Number of folds.

    Returns:
        list: One score per fold.
    """
    scores = pipeline.evaluate(scoring=scoring, cv=cv)
    logger.debug("Evaluation complete. Mean score: %.4f", np.mean(scores))
    return scores
