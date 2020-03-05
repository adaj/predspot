"""
Author: Adelson Araujo
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline


class FeatureScaling(TransformerMixin, BaseEstimator):

    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, x, y=None):
        self._estimator.fit(x,y)
        return self

    def transform(self, x):
        return pd.DataFrame(self._estimator.transform(x), index=x.index,
                            columns=x.columns)


class FeatureSelection(FeatureScaling):

    def transform(self, x):
        return pd.DataFrame(self._estimator.transform(x), index=x.index,
                            columns=x.columns[self._estimator.support_])


class Model(RegressorMixin, BaseEstimator):

    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, x, y=None):
        self._estimator.fit(x,y)
        return self

    def predict(self, x):
        return pd.DataFrame(self._estimator.predict(x), index=x.index,
                            columns=['crime_density'])


class PredictionPipeline(RegressorMixin, BaseEstimator):

    def __init__(self, mapping, fextraction, estimator):
        self._mapping = mapping
        self._fextraction = fextraction
        self._estimator = estimator
        self._dataset = None
        self._t_plus_one = None
        if mapping._tfreq == 'M':
            self._offset = pd.tseries.offsets.MonthEnd(1)
        elif mapping._tfreq == 'W':
            self._offset = pd.tseries.offsets.Week(1)
        elif mapping._tfreq == 'D':
            self._offset = pd.tseries.offsets.Day(1)

    def fit(self, dataset, y=None):
        self._dataset = dataset
        self._stseries = self._mapping.fit_transform(dataset.crimes)
        self._X = self._fextraction.fit_transform(self._stseries)
        t0 = self._X.index.get_level_values('t').unique().min()
        tf = self._stseries.index.get_level_values('t').unique().max()
        self._estimator.fit(self._X.loc[t0:tf], self._stseries.loc[t0:tf])
        self._t_plus_one = self._X.index.get_level_values('t').unique()[-1]
        return self

    def predict(self):
        X = self._X.loc[[self._t_plus_one],:]
        y_pred = self._estimator.predict(X)
        y_pred = pd.DataFrame(y_pred, index=X.index)
        self._stseries = self._stseries.append(y_pred['crime_density'])
        self._X = self._fextraction.transform(self._stseries)
        self._t_plus_one += self._offset
        return y_pred
