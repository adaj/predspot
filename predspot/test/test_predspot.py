import unittest
import sys
sys.path.append('../..')
sys.path.append('/Users/adelsondias/Documents/Repos/dev-predspot')

import numpy as np
import pandas as pd
import geopandas as gpd

from predspot.dataset_preparation import *
from predspot.feature_engineering import *
from predspot.ml_modelling import *
from predspot.utilities import *


(pd.to_datetime(start_time)+pd.tseries.offsets.MonthEnd(2)).strftime('%Y-%m-%d')




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
        self.grid_resolution = 100
        self.freq = 'M'
        self.lags = 2
        self.analysis_start_time = '2019-03-01'
        self.analysis_end_time = '2019-12-31'
        self.datasets, self.features = {}, {}
        for crime_type in crimes['tag'].unique():
            crime_data = crimes.loc[crimes['tag']==crime_type]
            print(f'setting up - {crime_type} ({len(crime_data)} samples) ...')
            self.datasets[crime_type] = Dataset(crime_data, study_area,
                                   self.grid_resolution,
                                   poi_data=None)

            kde = KDE(tfreq=self.freq,
                      grid=self.datasets[crime_type].grid,
                      start_time=self.analysis_start_time,
                      end_time=self.analysis_end_time)
            stseries = kde.fit_transform(self.datasets[crime_type].crimes)
            self.features[crime_type] = PandasFeatureUnion([
                        ('seasonal', Seasonality(lags=self.lags)),
                        ('trend', Trend(lags=self.lags)),
                        ('diff', Diff(lags=self.lags))
                        # geographic features...
            ]).fit_transform(stseries)

    def test_feature_engineering(self):
        for crime_type in self.datasets:
            print(f'testing - {crime_type}')
            self.assertEqual(self.features[crime_type].index.get_level_values('t').min(),
                             pd.to_datetime('2019-05-31'))
            self.assertEqual(self.features[crime_type].index.get_level_values('t').max(),
                             pd.to_datetime('2020-01-31'))


if __name__ == '__main__':
    unittest.main()


# crimes, study_area = generate_testdata(n_points=5000,
#                                        start_time='2019-01-01',
#                                        end_time='2020-01-01')
# dataset = Dataset(crimes, study_area, grid_resolution=100, poi_data=None)
# dataset
# %matplotlib inline
# dataset.plot(crime_samples=1000,
#              crimes={'markersize':1},
#              study_area={'linewidth':2})
