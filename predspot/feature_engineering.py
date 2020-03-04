import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline, _fit_transform_one, _transform_one
from scipy.stats import gaussian_kde
from pandas.io.json import json_normalize
from stldecompose import decompose
from joblib import Parallel, delayed
from scipy import sparse

import predspot.dataset_preparation

pd.options.mode.chained_assignment = None
idx = pd.IndexSlice

from abc import ABC, abstractmethod


#
#
# crime_tags = {'todas_ocorrencias': ['ASSALTO', 'OUTROS', 'AMEACAS', 'CONSUMO_DE_DROGAS',
#                                     'VEICULOS_ABERTOS', 'ACIDENTES', 'FURTO_DE_TERCEIROS']}
# g_resolution = 0.1
#
# city = gpd.read_file('ufrn/ufrn.geojson')
# cdata = pd.read_csv('ufrn/crime_data.csv', index_col=0)
#
# todas_ocs = cdata.loc[cdata['tag'].isin(crime_tags['todas_ocorrencias'])]
# dataset = predspot.dataset_preparation.Dataset(crimes=todas_ocs,
#                                                study_area=city,
#                                                grid_resolution=g_resolution)
# #dataset.plot()
#
# train, test = dataset.train_test_split(test_size=0.25)
#
# stseries = KDE(tfreq='M', grid=dataset.grid).transform(data_points=dataset.crimes)
# stseries.head()
#
# pipeline = Pipeline([
#     ('mapping', KDE(tfreq='M', grid=dataset.grid)),
#     ('feature_extraction', PandasFeatureUnion([
#         ('seasonal_trend', STLFeatures(lags=2, offset=1)),
#         ('diff', DiffFeatures(lags=2)) # geographic features, and so on...
#     ]))
# ])
# X = pipeline.fit_transform(dataset.crimes)
# X.head()
# #
# #
# #


class SpatioTemporalMapping(ABC, TransformerMixin, BaseEstimator):

    def __init__(self, tfreq, grid):
        self._tfreq = tfreq
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
        stseries = time_data_chunks.apply(self.fit_grid).to_frame('crime_density')
        stseries = json_normalize(data=stseries['crime_density'])
        stseries.index = time_data_chunks.index
        stseries = stseries.unstack()
        stseries.index.names = ['places','t']
        stseries = stseries.swaplevel().sort_index()
        return stseries


class KDE(SpatioTemporalMapping):

    def __init__(self, tfreq, grid, bw='silverman'):
        self._tfreq = tfreq
        self._grid = grid
        self._bw = bw
        self._kernel = None

    def fit_grid(self, data_points, as_df=False):
        if isinstance(self._bw, str):
            self._kernel = gaussian_kde(np.vstack([data_points.centroid.x,
                                                   data_points.centroid.y]),
                                        bw_method='silverman')
            self._bw = self._kernel.factor
        else:
            self._kernel = gaussian_kde(np.vstack([data_points.centroid.x,
                                                   data_points.centroid.y]),
                                        bw_method=self._bw)
        crime_density = pd.DataFrame(self._kernel(self._grid[['lon', 'lat']].values.T),
                                     index=self._grid.index, columns=['crime_density'])
        if as_df:
            return crime_density
        return crime_density.to_dict()['crime_density']


def make_lag_df(ts, lags, label):
    assert isinstance(lags, int) and len(ts) > lags, "lags parameter not allowed."
    lag_df = pd.concat([ts.shift(lag) for lag in range(1, lags+1)], axis=1).dropna()
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
        places = stseries.index.get_level_values('places').unique()
        for place in places:
            ts = stseries.loc[idx[:,place]]
            f, _ = make_lag_df(ts.diff()[1:], self._lags, label='diff')
            f['place'] = place
            f = f.set_index('place',append=True)
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
        places = stseries.index.get_level_values('places').unique()
        for place in places:
            ts = stseries.loc[idx[:,place]]
            f, _ = make_lag_df(ts, self._lags, label='ar')
            f['place'] = place
            f = f.set_index('place',append=True)
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
        places = stseries.index.get_level_values('places').unique()
        for place in places:
            ts = stseries.loc[idx[:,place]]
            stl = decompose(ts, period=self._lags)
            sx, _ = make_lag_df(stl.seasonal, self._lags, 'seasonal')
            tx, _ = make_lag_df(stl.trend, self._lags, 'trend')
            f = sx.join(tx)
            f['place'] = place
            f = f.set_index('place',append=True)
            X = X.append(f)
        X = X.sort_index()
        if self._offset is not None:
            X = X.loc[X.index.get_level_values('t').unique()[self._offset]:]
        return X


class FeatureUnion(FeatureUnion):
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
            return np.zeros((X.shape[0], 0))
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
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
