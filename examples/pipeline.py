"""
Pipeline Module

This module provides the main pipeline functionality for crime prediction,
including data loading, preprocessing, and model execution.

Example:
    >>> from predspot.pipeline import generate_testdata, run_prediction_pipeline
    >>> crime_data, study_area = generate_testdata(10000, '2020-01-01', '2020-12-31')
    >>> results = run_prediction_pipeline(crime_data, study_area)
"""

__author__ = 'Adelson Araujo'

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

from predspot import dataset_preparation
from predspot import crime_mapping
from predspot import feature_engineering
from predspot import ml_modelling
from predspot.utilities import PandasFeatureUnion


def generate_testdata(n_points, start_time, end_time, debug=False):
    """
    Generate synthetic crime data for testing.

    Args:
        n_points (int): Number of crime incidents to generate
        start_time (str): Start date in 'YYYY-MM-DD' format
        end_time (str): End date in 'YYYY-MM-DD' format
        debug (bool, optional): Enable debug printing. Defaults to False

    Returns:
        tuple: (crimes_df, study_area_gdf) - Generated crime data and study area

    Example:
        >>> crimes, area = generate_testdata(1000, '2020-01-01', '2020-12-31')
    """
    if debug:
        print(f"Generating {n_points} test data points from {start_time} to {end_time}", flush=True)

    study_area = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    study_area = study_area.loc[study_area['name']=='Brazil']

    crimes = pd.DataFrame()
    crime_types = pd.Series(['burglary', 'assault', 'drugs', 'homicide'])
    bounds = study_area.geometry.bounds.values[0]

    if debug:
        print("Generating random dates and locations", flush=True)

    def random_dates(start, end, n=10):
        """Generate random dates within a range."""
        start, end = pd.to_datetime(start), pd.to_datetime(end)
        start_u = start.value//10**9
        end_u = end.value//10**9
        return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

    crimes['tag'] = crime_types.sample(n_points, replace=True,
                                     weights=[1000, 100, 10, 1])
    crimes['t'] = random_dates(start_time, end_time, n_points)
    crimes['lat'] = np.random.uniform(bounds[1], bounds[3], n_points)
    crimes['lon'] = np.random.uniform(bounds[0], bounds[2], n_points)
    crimes.reset_index(drop=True, inplace=True)

    if debug:
        print("Test data generation complete", flush=True)

    return crimes, study_area


def run_prediction_pipeline(crime_data, study_area, crime_tags=None, time_range=None, 
                          tfreq='M', grid_resolution=250, debug=False):
    """
    Run the complete crime prediction pipeline.

    Args:
        crime_data (pandas.DataFrame): Crime incident data
        study_area (geopandas.GeoDataFrame): Study area boundaries
        crime_tags (list, optional): List of crime types to include
        time_range (list, optional): Time range as ['HH:MM', 'HH:MM']
        tfreq (str, optional): Time frequency ('M', 'W', 'D'). Defaults to 'M'
        grid_resolution (float, optional): Spatial grid resolution in km. Defaults to 250
        debug (bool, optional): Enable debug printing. Defaults to False

    Returns:
        tuple: (predictions, pipeline) - Predicted crime densities and fitted pipeline

    Raises:
        ValueError: If input data is invalid or missing required columns
    """
    if debug:
        print("Initializing prediction pipeline", flush=True)

    # Validate input data
    required_columns = ['tag', 't', 'lat', 'lon']
    if not all(col in crime_data.columns for col in required_columns):
        raise ValueError(f"Crime data must contain columns: {required_columns}")

    # Filter by crime tags if specified
    if crime_tags:
        if debug:
            print(f"Filtering for crime types: {crime_tags}", flush=True)
        crime_data = crime_data.loc[crime_data['tag'].isin(crime_tags)]

    # Filter by time range if specified
    if time_range:
        if debug:
            print(f"Filtering for time range: {time_range}", flush=True)
        time_ix = pd.DatetimeIndex(crime_data['t'])
        crime_data = crime_data.iloc[time_ix.indexer_between_time(time_range[0], time_range[1])]

    if debug:
        print("Creating dataset", flush=True)
    dataset = dataset_preparation.Dataset(crimes=crime_data, study_area=study_area)

    if debug:
        print("Building prediction pipeline", flush=True)
    pred_pipeline = ml_modelling.PredictionPipeline(
        mapping=crime_mapping.KDE(
            tfreq=tfreq,
            bandwidth='auto',
            grid=crime_mapping.create_gridpoints(study_area, grid_resolution)
        ),
        fextraction=PandasFeatureUnion([
            ('seasonal', feature_engineering.Seasonality(lags=2)),
            ('trend', feature_engineering.Trend(lags=2)),
            ('diff', feature_engineering.Diff(lags=2))
        ]),
        estimator=Pipeline([
            ('f_scaling', feature_engineering.FeatureScaling(
                QuantileTransformer(10, output_distribution='uniform'))),
            ('f_selection', ml_modelling.FeatureSelection(
                RFE(RandomForestRegressor()))),
            ('model', ml_modelling.Model(RandomForestRegressor(n_estimators=50)))
        ])
    )

    if debug:
        print("Fitting pipeline", flush=True)
    pred_pipeline.fit(dataset)

    if debug:
        print("Making predictions", flush=True)
    predictions = pred_pipeline.predict()

    if debug:
        print("Pipeline execution complete", flush=True)

    return predictions, pred_pipeline


def evaluate_pipeline(pipeline, scoring='r2', cv=5, debug=False):
    """
    Evaluate the prediction pipeline using cross-validation.

    Args:
        pipeline (PredictionPipeline): Fitted prediction pipeline
        scoring (str, optional): Scoring metric ('r2' or 'mse'). Defaults to 'r2'
        cv (int, optional): Number of cross-validation folds. Defaults to 5
        debug (bool, optional): Enable debug printing. Defaults to False

    Returns:
        list: Cross-validation scores

    Example:
        >>> scores = evaluate_pipeline(fitted_pipeline, scoring='r2', cv=5)
    """
    if debug:
        print(f"Evaluating pipeline with {cv}-fold CV using {scoring} metric", flush=True)

    scores = pipeline.evaluate(scoring=scoring, cv=cv)

    if debug:
        print(f"Evaluation complete. Mean score: {np.mean(scores):.4f}", flush=True)

    return scores 