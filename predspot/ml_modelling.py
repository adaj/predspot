"""
Supervised Learning Modelling
"""

__author__ = 'Adelson Araujo'

import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

idx = pd.IndexSlice


class FeatureSelection(TransformerMixin, BaseEstimator):

    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, x, y=None):
        self._estimator.fit(x,y)
        return self

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
        self._stseries = None
        self._dataset = None
        if mapping._tfreq == 'M':
            self._offset = pd.tseries.offsets.MonthEnd(1)
        elif mapping._tfreq == 'W':
            self._offset = pd.tseries.offsets.Week(1)
        elif mapping._tfreq == 'D':
            self._offset = pd.tseries.offsets.Day(1)

    @property
    def grid(self):
        return self._mapping._grid

    @property
    def dataset(self):
        return self._dataset

    @property
    def stseries(self):
        return self._stseries

    @property
    def feature_importances(self):
        # TODO: this needs improvements
        assert self._X is not None, 'this instance was not fitted yet.'
        try:
            return pd.DataFrame(self._estimator.steps[-1][1]._estimator.feature_importances_,
                                index=self._X.columns[self._estimator.steps[-2][1]._estimator.support_],
                                columns=['importance'])
        except:
            raise Exception('estimator used has not feature importances implemented yet.')

    def evaluate(self, scoring, cv=5):
        assert self._X is not None, 'this instance was not fitted yet.'
        if scoring == 'r2':
            scoring = r2_score
        elif scoring == 'mse':
            scoring = mean_squared_error
        else:
            raise Exception('invalid scoring. Try "r2" or "mse".')
        timestamps = self._X.index.get_level_values('t').unique()\
            .intersection(self._stseries.index.get_level_values('t').unique())
        assert isinstance(cv, int) and cv < len(timestamps), \
            'cv must be an integer and not higher than the number of timestamps available.'
        scores = []
        for train_t, test_t in TimeSeriesSplit(cv).split(timestamps):
            X_train = self._X.loc[idx[timestamps[train_t], :], :].sample(frac=1) # shuffle for training
            X_test = self._X.loc[idx[timestamps[test_t], :], :]
            y_train = self._stseries.loc[X_train.index]
            y_test = self._stseries.loc[timestamps[test_t]]
            self._estimator.fit(X_train, y_train)
            y_pred = self._estimator.predict(X_test)
            scores.append(scoring(y_test, y_pred))
        self.fit(self._dataset) # back to normal
        return scores

    def fit(self, dataset, y=None):
        self._dataset = dataset
        self._stseries = self._mapping.fit_transform(dataset.crimes)
        self._X = self._fextraction.fit_transform(self._stseries)
        t0 = self._X.index.get_level_values('t').unique().min()
        tf = self._stseries.index.get_level_values('t').unique().max()
        X = self._X.loc[t0:tf].sample(frac=1) # shuffle for training
        y = self._stseries.loc[X.index]       # shuffle for training
        try:
            self._estimator.fit(X, y)
        except:
            raise Exception(f'ERRO: {t0}, {tf} \nX: {self._X.loc[t0:tf]}')
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
