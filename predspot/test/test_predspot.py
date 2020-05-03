import unittest
import sys

import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

from predspot.dataset_preparation import *
from predspot.crime_mapping import *
from predspot.feature_engineering import *
from predspot.ml_modelling import *
from predspot.utilities import *


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


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        crimes, study_area = generate_testdata(n_points=10000,
                                               start_time='2019-01-31',
                                               end_time='2019-12-31')
        grid = create_gridpoints(study_area, resolution=100)
        freq = 'M'
        lags = 2
        # analysis_start_time = '2019-03-01'
        # analysis_end_time = '2019-12-31'
        self.pipelines = []
        for crime_type in crimes['tag'].unique():
            crime_data = crimes.loc[crimes['tag']==crime_type]
            print(f'setting up - {crime_type} ({len(crime_data)} samples) ...')
            dataset = Dataset(crimes=crime_data, study_area=study_area)
            pred_pipeline = PredictionPipeline(
                mapping = KDE(tfreq = freq, bandwidth='auto', grid = grid),
                fextraction=PandasFeatureUnion([
                    ('seasonal', Seasonality(lags)),
                    ('trend', Trend(lags)),
                    ('diff', Diff(lags))
                    # geographic features
                ]),
                estimator=Pipeline([
                    ('f_scaling', FeatureScaling(
                            QuantileTransformer(10, output_distribution='uniform'))),
                    ('f_selection', FeatureSelection(
                            RFE(RandomForestRegressor()))),
                    ('model', Model(RandomForestRegressor(n_estimators=50)))
                ])
            )
            self.pipelines.append(pred_pipeline.fit(dataset))

    # def test_feature_engineering(self):
    #     for p in self.pipelines:
    #         self.assertEqual(p._X.index.get_level_values('t').min(),
    #                          pd.to_datetime('2019-04-30'))
    #         self.assertEqual(p._X.index.get_level_values('t').max(),
    #                          pd.to_datetime('2020-01-31'))

    def test_evaluation(self):
        for p in self.pipelines:
            p.evaluate('r2')
            break


if __name__ == '__main__':
    unittest.main()
