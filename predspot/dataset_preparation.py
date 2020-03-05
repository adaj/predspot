"""
Author: Adelson Araujo
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


class Dataset:

    def __init__(self, crimes, study_area, grid_resolution=0.5, poi_data=None):
        if isinstance(crimes, pd.DataFrame):
            assert all([attribute in crimes.columns for attribute in ['tag', 't', 'lon', 'lat']]),\
                "Input crime data must have at least ['tag', 't', 'lon', 'lat'] as columns."
            self._crimes = crimes
            self._crimes['geometry'] = self._crimes.apply(lambda x: Point([x['lon'], x['lat']]),
                                                          axis=1)
            self._crimes = gpd.GeoDataFrame(self._crimes, crs={'init': 'epsg:4326'})
            self._crimes['t'] = self._crimes['t'].apply(pd.to_datetime)
        else:
            raise Exception("crimes must be in DataFrame format.")
        if isinstance(study_area, gpd.GeoDataFrame):
            self._study_area = study_area
        else:
            raise Exception("study_area must be in GeoDataFrame format.")
        if isinstance(grid_resolution, float) and grid_resolution > 0:
            self._grid_resolution = grid_resolution
            self._grid, self.lonv, self.latv = make_gridpoints(self._study_area,
                                                               grid_resolution, True)
        else:
            raise Exception("Invalid grid_resolution.")
        self._poi_data = poi_data

    @property
    def crimes(self):
        return self._crimes

    @property
    def study_area(self):
        return self._study_area

    @property
    def grid(self):
        return self._grid

    @property
    def poi_data(self):
        return self._poi_data

    def plot(self, ax=None):
        if ax is None:
            ax = self.study_area.plot(color='white', edgecolor='black')
        else:
            self.study_area.plot(color='white', edgecolor='black', ax=ax)
        self.grid.plot(color='orange', ax=ax, markersize=2)
        self.crimes.plot(color='red', ax=ax, marker='x')
        return ax

    def train_test_split(self, test_size=0.25):
        test_dataset = Dataset(self.crimes.sample(frac=test_size),
                               self.study_area, self._grid_resolution)
        train_ix = set(self.crimes.index) - set(test_dataset.crimes.index)
        train_dataset = Dataset(self.crimes.loc[train_ix],
                                self.study_area, self._grid_resolution)
        return train_dataset, test_dataset


def make_gridpoints(study_area, resolution=1, return_coords=False):
    bounds = study_area.bounds
    b_s, b_w = bounds.min().values[1], bounds.min().values[0]
    b_n, b_e = bounds.max().values[3], bounds.max().values[2]
    nlon = int(np.ceil((b_e-b_w) / (resolution/111.32)))
    nlat = int(np.ceil((b_n-b_s) / (resolution/110.57)))
    lonv, latv = np.meshgrid(np.linspace(b_w, b_e, nlon), np.linspace(b_s, b_n, nlat))
    gridpoints = pd.DataFrame(np.vstack([lonv.ravel(), latv.ravel()]).T,
                              columns=['lon', 'lat'])
    gridpoints['geometry'] = gridpoints.apply(lambda x: Point([x['lon'], x['lat']]),
                                              axis=1)
    gridpoints = gpd.GeoDataFrame(gridpoints, crs={'init': 'epsg:4326'})
    grid_ix = gpd.sjoin(gridpoints, study_area, op='intersects').index.unique()
    if len(grid_ix) == 0:
        raise Exception("Resolution too coarse to return a useful grid, " \
                        + "Try smaller values.")
    gridpoints = gridpoints.loc[grid_ix]
    gridpoints.index.name = 'places'
    if return_coords:
        return gridpoints, lonv, latv
    return gridpoints
