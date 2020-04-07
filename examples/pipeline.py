"""
Author: Adelson Araujo
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

from predspot import dataset_preparation
from predspot import crime_mapping
from predspot import feature_engineering
from predspot import ml_modelling
from predspot.utilities import PandasFeatureUnion


def generate_testdata(n_points, start_time, end_time):
    study_area = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    study_area = study_area.loc[study_area['name']=='Brazil']

    crimes = pd.DataFrame()
    crime_types = pd.Series(['burglary', 'assault', 'drugs', 'homicide'])
    bounds = study_area.geometry.bounds.values[0]
    def random_dates(start, end, n=10):
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
    return crimes, study_area

# Loading data: randomly distributed crimes in Brazil (whole country)
crime_data, study_area = generate_testdata(n_points=10000,
                                            start_time='2020-01-01',
                                            end_time='2020-12-31')


# Filtering by crime_tags
crime_tags = ['burglary', 'assault']
cdata = crime_data.loc[crime_data['tag'].isin(crime_tags)]


# Filtering between_time
between_time = ['6:00', '18:00']
time_ix = pd.DatetimeIndex(cdata['t'])
cdata = cdata.iloc[time_ix.indexer_between_time(between_time[0],between_time[1])]

# Prediction settings
tfreq = 'M' # monthly predictions
grid_resolution = 250 # km

# Use Predspot
dataset = dataset_preparation.Dataset(crimes=cdata, study_area=study_area)
print('Dataset ready!')

pred_pipeline = ml_modelling.PredictionPipeline(
    mapping = crime_mapping.KDE(tfreq = tfreq, bandwidth='auto',
                  grid = crime_mapping.create_gridpoints(study_area, grid_resolution)),
    fextraction=PandasFeatureUnion([
        ('seasonal', feature_engineering.Seasonality(lags=2)),
        ('trend', feature_engineering.Trend(lags=2)),
        ('diff', feature_engineering.Diff(lags=2))
        # geographic features
    ]),
    estimator=Pipeline([
        ('f_scaling', feature_engineering.FeatureScaling(
                            QuantileTransformer(10, output_distribution='uniform'))),
        ('f_selection', ml_modelling.FeatureSelection(
                            RFE(RandomForestRegressor()))),
        ('model', ml_modelling.Model(RandomForestRegressor(n_estimators=50)))
    ])
)
pred_pipeline.fit(dataset)
print('Pipeline adjusted!')

# Now you can predict for the next month
y_pred = pred_pipeline.predict()
