"""
Utilities Module

This module provides utility functions and classes for data processing and visualization,
including contour generation, feature union operations, and pandas-specific transformations.
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
import geojsoncontour


def contour_geojson(y, bbox, resolution, cmin, cmax, debug=False):
    """
    Generate GeoJSON contours from spatial data.

    Args:
        y (pandas.Series): Values to contour
        bbox (GeoDataFrame): Bounding box for the contour
        resolution (float): Spatial resolution in kilometers
        cmin (float): Minimum contour value
        cmax (float): Maximum contour value
        debug (bool, optional): Enable debug printing. Defaults to False

    Returns:
        dict: GeoJSON representation of the contours

    Raises:
        AssertionError: If bbox is not a GeoDataFrame
    """
    if debug:
        print(f"Generating contours with resolution: {resolution}km", flush=True)
        
    assert isinstance(bbox, GeoDataFrame)
    bounds = bbox.bounds
    b_s, b_w = bounds.min().values[1], bounds.min().values[0]
    b_n, b_e = bounds.max().values[3], bounds.max().values[2]
    
    # Calculate grid dimensions
    nlon = int(ceil((b_e-b_w) / (resolution/111.32)))
    nlat = int(ceil((b_n-b_s) / (resolution/110.57)))
    
    if debug:
        print(f"Grid dimensions: {nlon}x{nlat}", flush=True)
    
    # Create meshgrid and initialize values
    lonv, latv = meshgrid(linspace(b_w, b_e, nlon), linspace(b_s, b_n, nlat))
    Z = zeros(lonv.shape[0]*lonv.shape[1]) - 999
    Z[y.index] = y.values
    Z = Z.reshape(lonv.shape)
    
    # Generate contours
    fig, axes = plt.subplots()
    contourf = axes.contourf(lonv, latv, Z,
                            levels=linspace(cmin, cmax, 25),
                            cmap='Spectral_r')
    
    if debug:
        print("Converting contours to GeoJSON", flush=True)
        
    geojson = geojsoncontour.contourf_to_geojson(contourf=contourf, fill_opacity=0.5)
    plt.close(fig)
    return geojson


class PandasFeatureUnion(FeatureUnion):
    """
    A FeatureUnion transformer that preserves pandas DataFrames.
    
    This class extends sklearn's FeatureUnion to work with pandas DataFrames,
    maintaining index alignment and column names.

    Attributes:
        n_jobs (int): Number of parallel jobs
        transformer_list (list): List of transformer tuples
        transformer_weights (dict): Weights for transformers
        debug (bool): Enable debug printing
    """

    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, debug=False):
        super().__init__(transformer_list, n_jobs, transformer_weights)
        self.debug = debug
        
        if self.debug:
            print("Initializing PandasFeatureUnion", flush=True)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit all transformers and transform the data.

        Args:
            X (pandas.DataFrame): Input features
            y (array-like, optional): Target values
            **fit_params: Additional fitting parameters

        Returns:
            pandas.DataFrame: Transformed features

        Raises:
            ValueError: If no transformers are provided
        """
        if self.debug:
            print(f"Fitting and transforming {len(X)} samples", flush=True)
            
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
        
        if self.debug:
            print("Merging transformed features", flush=True)
            
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        """
        Merge transformed features into a single DataFrame.

        Args:
            Xs (list): List of transformed DataFrames

        Returns:
            pandas.DataFrame: Merged DataFrame
        """
        if self.debug:
            print(f"Merging {len(Xs)} DataFrames", flush=True)
            
        return pd.concat(Xs, axis="columns", copy=False).dropna()

    def transform(self, X):
        """
        Transform X separately by each transformer.

        Args:
            X (pandas.DataFrame): Input features

        Returns:
            pandas.DataFrame: Transformed features
        """
        if self.debug:
            print(f"Transforming features with {len(self.transformer_list)} transformers", flush=True)
            
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
