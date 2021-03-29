"""
Predspot utilities
"""

__author__ = 'Adelson Araujo'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from numpy import zeros, linspace, ceil, meshgrid
from sklearn.pipeline import FeatureUnion, Pipeline, _fit_transform_one, _transform_one
from joblib import Parallel, delayed
from scipy import sparse


def contour_geojson(y, bbox, resolution, cmin, cmax):
    import geojsoncontour
    assert isinstance(bbox, GeoDataFrame)
    bounds = bbox.bounds
    b_s, b_w = bounds.min().values[1], bounds.min().values[0]
    b_n, b_e = bounds.max().values[3], bounds.max().values[2]
    nlon = int(ceil((b_e-b_w) / (resolution/111.32)))
    nlat = int(ceil((b_n-b_s) / (resolution/110.57)))
    lonv, latv = meshgrid(linspace(b_w, b_e, nlon), linspace(b_s, b_n, nlat))
    Z = zeros(lonv.shape[0]*lonv.shape[1]) - 999
    Z[y.index] = y.values
    Z = Z.reshape(lonv.shape)
    fig, axes = plt.subplots()
    contourf = axes.contourf(lonv, latv, Z,
                            levels=linspace(cmin, cmax, 25),
                            cmap='Spectral_r')
    geojson = geojsoncontour.contourf_to_geojson(contourf=contourf, fill_opacity=0.5)
    plt.close(fig)
    return geojson


class PandasFeatureUnion(FeatureUnion):
    """Adapted from:
    https://github.com/marrrcin/pandas-feature-union/blob/master/pandas_feature_union.py
    """
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
        return pd.concat(Xs, axis="columns", copy=False).dropna()

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
