"""
Utilities Module
================

Helpers used across Predspot: a :class:`PandasFeatureUnion` that keeps
DataFrames (and their index) when combining feature transformers, and a
GeoJSON contour export for density maps.
"""

__author__ = 'Adelson Araujo'

import logging

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class PandasFeatureUnion(TransformerMixin, BaseEstimator):
    """
    Concatenate the DataFrame outputs of several transformers column-wise.

    Unlike :class:`sklearn.pipeline.FeatureUnion`, the transformers' outputs
    are aligned on their index and returned as a DataFrame. Rows with missing
    values after alignment (e.g. warm-up rows of lag features) are dropped.

    Args:
        transformer_list (list): ``(name, transformer)`` pairs.
    """

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def _iter(self):
        for name, transformer in self.transformer_list:
            if transformer is None or transformer == 'drop':
                continue
            yield name, transformer

    def fit(self, X, y=None, **fit_params):
        for _, transformer in self._iter():
            transformer.fit(X, y, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        outputs = [transformer.fit_transform(X, y, **fit_params)
                   for _, transformer in self._iter()]
        return self.merge_dataframes_by_column(outputs)

    def transform(self, X):
        outputs = [transformer.transform(X) for _, transformer in self._iter()]
        return self.merge_dataframes_by_column(outputs)

    @staticmethod
    def merge_dataframes_by_column(outputs):
        """
        Align a list of DataFrames on their index and concatenate columns.

        Args:
            outputs (list): DataFrames returned by the transformers.

        Returns:
            pandas.DataFrame: The merged features without missing rows.
        """
        if not outputs:
            raise ValueError('PandasFeatureUnion has no transformers.')
        logger.debug('Merging %d feature blocks', len(outputs))
        return pd.concat(outputs, axis='columns').dropna()


def contour_geojson(y, bbox, resolution, cmin, cmax):
    """
    Export a density surface as filled GeoJSON contours.

    Requires the optional dependency ``geojsoncontour``
    (``pip install predspot[contour]``).

    Args:
        y (pandas.Series): Values indexed by the positional index of the
            full point grid returned by
            :func:`predspot.crime_mapping.create_gridpoints` (before
            clipping), i.e. the ``places`` index.
        bbox (GeoDataFrame): Study area used to build the grid.
        resolution (float): Grid resolution in kilometers (same as the grid).
        cmin (float): Lowest contour level.
        cmax (float): Highest contour level.

    Returns:
        str: GeoJSON string with the contour polygons.
    """
    try:
        import geojsoncontour
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('contour_geojson requires the optional dependency '
                          '`geojsoncontour`: pip install predspot[contour]') from exc
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from predspot.crime_mapping import (KM_PER_DEG_LAT, KM_PER_DEG_LON,
                                        _check_bbox, _wgs84_bounds)

    _check_bbox(bbox)
    b_w, b_s, b_e, b_n = _wgs84_bounds(bbox)
    nlon = max(int(np.ceil((b_e - b_w) / (resolution / KM_PER_DEG_LON))), 2)
    nlat = max(int(np.ceil((b_n - b_s) / (resolution / KM_PER_DEG_LAT))), 2)
    lonv, latv = np.meshgrid(np.linspace(b_w, b_e, nlon), np.linspace(b_s, b_n, nlat))
    Z = np.full(lonv.size, -999.0)
    Z[np.asarray(y.index, dtype=int)] = y.values
    Z = Z.reshape(lonv.shape)

    fig, axes = plt.subplots()
    contourf = axes.contourf(lonv, latv, Z, levels=np.linspace(cmin, cmax, 25),
                             cmap='Spectral_r')
    geojson = geojsoncontour.contourf_to_geojson(contourf=contourf, fill_opacity=0.5)
    plt.close(fig)
    return geojson
