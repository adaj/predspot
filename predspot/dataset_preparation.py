"""
Dataset Preparation
"""

__author__ = 'Adelson Araujo'

import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
from shapely.geometry import Point, LineString


class Dataset:

    def __init__(self, crimes, study_area, poi_data=None):
        assert isinstance(study_area, gpd.GeoDataFrame), \
            "study_area must be a geopandas GeoDataFrame."
        self._study_area = study_area
        assert isinstance(crimes, pd.DataFrame) \
               and all([x in crimes.columns for x in ['tag', 't', 'lon', 'lat']]),\
            "Input crime data must be a pandas Data Frame and " \
            + "have at least `tag`, `t`, `lon` and `lat` as columns."
        self._crimes = crimes
        self._crimes['geometry'] = self._crimes.apply(lambda x: Point([x['lon'], x['lat']]),
                                                      axis=1)
        self._crimes = gpd.GeoDataFrame(self._crimes, crs={'init': 'epsg:4326'})
        self._crimes['t'] = self._crimes['t'].apply(pd.to_datetime)
        if poi_data is None:
            self._poi_data = gpd.GeoDataFrame()
        else:
            self._poi_data = poi_data

    def __repr__(self):
        return 'predspot.Dataset<\n'\
             + f'  crimes = GeoDataFrame({self._crimes.shape[0]}),\n' \
             + f'    >> {self.crimes["tag"].value_counts().to_dict()}\n' \
             + f'  study_area = GeoDataFrame({self._study_area.shape[0]}),\n' \
             + f'  poi_data = GeoDataFrame({self._poi_data.shape[0]})\n' \
             + '>'

    @property
    def crimes(self):
        return self._crimes

    @property
    def study_area(self):
        return self._study_area

    @property
    def poi_data(self):
        return self._poi_data

    @property
    def shape(self):
        return {'crimes': self._crimes.shape,
                'study_area': self._study_area.shape,
                'poi_data': self._poi_data.shape}

    def plot(self, ax=None, crime_samples=1000, **kwargs):
        if ax is None:
            ax = self.study_area.plot(color='white', edgecolor='black',
                                      **kwargs.pop('study_area',{}))
        else:
            self.study_area.plot(color='white', edgecolor='black', ax=ax,
                                 **kwargs.pop('study_area',{}))
        if crime_samples > len(self.crimes):
            crime_samples = len(self.crimes)
        self.crimes.sample(crime_samples).plot(ax=ax, marker='x',
                                               **kwargs.pop('crimes',{}))
        return ax

    def train_test_split(self, test_size=0.25):
        assert 0 < test_size < 1, \
            'test_size must be between 0 and 1.'
        test_dataset = Dataset(self.crimes.sample(frac=test_size), self.study_area)
        train_ix = set(self.crimes.index) - set(test_dataset.crimes.index)
        train_dataset = Dataset(self.crimes.loc[train_ix], self.study_area)
        return train_dataset, test_dataset
