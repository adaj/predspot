"""
Feature Engineering
"""

__author__ = 'Adelson Araujo'

from abc import abstractmethod
from numpy import vstack
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from stldecompose import decompose


class TimeSeriesFeatures(BaseEstimator, TransformerMixin): # X

    def __init__(self, lags, tfreq):
        assert isinstance(lags, int) and lags > 1, \
            '`lags` must be a positive integer.'
        self._lags = lags
        assert tfreq in ['D', 'W', 'M'], \
            '`tfreq` not allowed, choose between `D`, `W`, `M`.'
        self._tfreq = tfreq

    @property
    def lags(self):
        return self._lags

    @property
    def label(self):
        return 'feature' # override this label if implementing a new feature

    @abstractmethod
    def apply_ts_decomposition(self, ts):
        pass

    def make_lag_df(self, ts):
        assert len(ts) > self.lags, "`lags` are higher than temporal units."
        lag_df = pd.concat([ts.shift(lag) for lag in range(1, self.lags+1)], axis=1)
        lag_df = lag_df.iloc[self.lags:]
        lag_df.columns = ['{}_{}'.format(self.label, i) for i in range(1, self.lags+1)]
        return lag_df, ts.loc[lag_df.index]

    def fit(self, x, y=None):
        return self

    def transform(self, stseries):
        X = pd.DataFrame()
        # freq = stseries.index.get_level_values('t').unique().inferred_freq
        if self._tfreq=='M':
            next_time = pd.tseries.offsets.MonthEnd(1)
        elif self._tfreq=='W':
            next_time = pd.tseries.offsets.Week(1)
        elif self._tfreq=='D':
            next_time = pd.tseries.offsets.Day(1)
        places = stseries.index.get_level_values('places').unique()
        for place in places:
            ts = stseries.loc[pd.IndexSlice[:, place]]
            ts = self.apply_ts_decomposition(ts)
            ts.loc[ts.index[-1] + next_time] = None
            f, _ = self.make_lag_df(ts)
            f['places'] = place
            f = f.set_index('places',append=True)
            X = X.append(f)
        X = X.sort_index()
        # if self._offset is not None:
        #     X = X.loc[X.index.get_level_values('t').unique()[self._offset]:]
        return X


class LatLonFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, grid):
        self.grid = grid

    def fit(self, x, y=None):
        return self

    def transform(self, stseries):
        return stseries.to_frame().join(self.grid[['lon', 'lat']]).drop(columns=0)


class AR(TimeSeriesFeatures):

    @property
    def label(self):
        return 'ar'

    def apply_ts_decomposition(self, ts):
        return ts


class Diff(TimeSeriesFeatures):

    @property
    def label(self):
        return 'diff'

    def apply_ts_decomposition(self, ts):
        return ts.diff()[1:]


class Seasonality(TimeSeriesFeatures):

    @property
    def label(self):
        return 'seasonal'

    def apply_ts_decomposition(self, ts):
        return decompose(ts, period=self._lags).seasonal


class Trend(TimeSeriesFeatures):

    @property
    def label(self):
        return 'trend'

    def apply_ts_decomposition(self, ts):
        return decompose(ts, period=self._lags).trend


class FeatureScaling(TransformerMixin, BaseEstimator):

    def __init__(self, estimator):
        self._estimator = estimator

    def fit(self, x, y=None):
        self._estimator.fit(x,y)
        return self

    def transform(self, x):
        return pd.DataFrame(self._estimator.transform(x), index=x.index,
                            columns=x.columns)
