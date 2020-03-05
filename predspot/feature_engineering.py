"""
Author: Adelson Araujo
"""

import pandas as pd
from numpy import vstack, zeros
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline, _fit_transform_one, _transform_one
from scipy.stats import gaussian_kde
from pandas.io.json import json_normalize
from stldecompose import decompose
from joblib import Parallel, delayed
from scipy import sparse

pd.options.mode.chained_assignment = None
idx = pd.IndexSlice


class SpatioTemporalMapping(ABC, TransformerMixin, BaseEstimator):

    def __init__(self, tfreq, grid):
        assert tfreq.upper() in ['M', 'W', 'D'], "Invalid tfreq. Please choose "\
                                                 + "(m)onthly, (w)eekly or (d)aily."
        self._tfreq = tfreq.upper()
        self._grid = grid

    @abstractmethod
    def fit_grid(self, data_points=None):
        pass

    def fit(self, x, y=None):
        return self

    def transform(self, data_points):
        time_data_chunks = data_points.set_index('t').resample(self._tfreq)
        time_data_chunks = pd.DataFrame(time_data_chunks,
                              columns=['t', 'crime_chunks'])
        time_data_chunks = time_data_chunks.set_index('t').sort_values('t')
        time_data_chunks = time_data_chunks.apply(lambda x: x[0],axis=1)
        try:
            stseries = time_data_chunks.apply(self.fit_grid).to_frame('crime_density')
        except:
            print(time_data_chunks.shape, time_data_chunks.apply(len), time_data_chunks)
            for ix in range(len(time_data_chunks.index)):
                print(ix)
                self.fit_grid(time_data_chunks.iloc[ix])

        stseries = json_normalize(data=stseries['crime_density'])
        stseries.index = time_data_chunks.index
        stseries = stseries.unstack()
        stseries.index.names = ['places','t']
        stseries = stseries.swaplevel().sort_index()
        return stseries


class KDE(SpatioTemporalMapping):

    def __init__(self, tfreq, grid, bw='silverman'):
        assert tfreq.upper() in ['M', 'W', 'D'], "Invalid tfreq. Please choose "\
                                                 + "(m)onthly, (w)eekly or (d)aily."
        self._tfreq = tfreq.upper()
        self._grid = grid
        self._bw = bw
        self._kernel = None

    def fit_grid(self, data_points, as_df=False):
        if len(data_points) < 3:
            crime_density = pd.DataFrame([0]*len(self._grid.index),
                                         index=self._grid.index, columns=['crime_density'])
        else:
            if isinstance(self._bw, str):
                self._kernel = gaussian_kde(vstack([data_points.centroid.x,
                                                       data_points.centroid.y]),
                                            bw_method='silverman')
                self._bw = self._kernel.factor
            else:
                self._kernel = gaussian_kde(vstack([data_points.centroid.x,
                                                       data_points.centroid.y]),
                                            bw_method=self._bw)
            crime_density = pd.DataFrame(self._kernel(self._grid[['lon', 'lat']].values.T),
                                         index=self._grid.index, columns=['crime_density'])
        if as_df:
            return crime_density
        return crime_density.to_dict()['crime_density']


def make_lag_df(ts, lags, label):
    assert isinstance(lags, int) and len(ts) > lags, "lags parameter not allowed."
    lag_df = pd.concat([ts.shift(lag) for lag in range(1, lags+1)], axis=1).iloc[lags:]
    lag_df.columns = ['{}_{}'.format(label, i) for i in range(1, lags+1)]
    return lag_df, ts.loc[lag_df.index]


class DiffFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, lags, offset=None):
        self._lags = lags
        self._offset = offset

    def fit(self, x, y=None):
        return self

    def transform(self, stseries):
        X = pd.DataFrame()
        freq = stseries.index.get_level_values('t').unique().inferred_freq
        if freq=='M':
            next_time = pd.tseries.offsets.MonthEnd(1)
        elif freq=='W':
            next_time = pd.tseries.offsets.Week(1)
        elif freq=='D':
            next_time = pd.tseries.offsets.Day(1)
        places = stseries.index.get_level_values('places').unique()
        for place in places:
            ts = stseries.loc[idx[:, place]]
            ts = ts.diff()[1:]
            t_plus_one = ts.index[-1] + next_time
            ts.loc[t_plus_one] = None
            f, _ = make_lag_df(ts, self._lags, label='diff')
            f['places'] = place
            f = f.set_index('places',append=True)
            X = X.append(f)
        X = X.sort_index()
        if self._offset is not None:
            X = X.loc[X.index.get_level_values('t').unique()[self._offset]:]
        return X


class AutoRegressiveFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, lags, offset=None):
        self._lags = lags
        self._offset = offset

    def fit(self, x, y=None):
        return self

    def transform(self, stseries):
        X = pd.DataFrame()
        freq = stseries.index.get_level_values('t').unique().inferred_freq
        if freq=='M':
            next_time = pd.tseries.offsets.MonthEnd(1)
        elif freq=='W':
            next_time = pd.tseries.offsets.Week(1)
        elif freq=='D':
            next_time = pd.tseries.offsets.Day(1)
        places = stseries.index.get_level_values('places').unique()
        for place in places:
            ts = stseries.loc[idx[:, place]]
            t_plus_one = ts.index[-1] + next_time
            ts.loc[t_plus_one] = None
            f, _ = make_lag_df(ts, self._lags, label='ar')
            f['places'] = place
            f = f.set_index('placess',append=True)
            X = X.append(f)
        X = X.sort_index()
        if self._offset is not None:
            X = X.loc[X.index.get_level_values('t').unique()[self._offset]:]
        return X


class STLFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, lags, offset=None):
        self._lags = lags
        self._offset = offset

    def fit(self, x, y=None):
        return self

    def transform(self, stseries):
        X = pd.DataFrame()
        freq = stseries.index.get_level_values('t').unique().inferred_freq
        if freq=='M':
            next_time = pd.tseries.offsets.MonthEnd(1)
        elif freq=='W':
            next_time = pd.tseries.offsets.Week(1)
        elif freq=='D':
            next_time = pd.tseries.offsets.Day(1)
        places = stseries.index.get_level_values('places').unique()
        for place in places:
            ts = stseries.loc[idx[:, place]]
            stl = decompose(ts, period=self._lags)
            t_plus_one = stl.seasonal.index[-1] + next_time
            stl.seasonal.loc[t_plus_one] = None
            stl.trend.loc[t_plus_one] = None
            sx, _ = make_lag_df(stl.seasonal, self._lags, 'seasonal')
            tx, _ = make_lag_df(stl.trend, self._lags, 'trend')
            f = sx.join(tx)
            f['places'] = place
            f = f.set_index('places',append=True)
            X = X.append(f)
        X = X.sort_index()
        if self._offset is not None:
            X = X.loc[X.index.get_level_values('t').unique()[self._offset]:]
        return X


class PandasFeatureUnion(FeatureUnion):
    # https://github.com/marrrcin/pandas-feature-union/blob/master/pandas_feature_union.py

    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
