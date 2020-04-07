"""
Crime Mapping
"""
__author__ = 'Adelson Araujo'
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.base import BaseEstimator, TransformerMixin
from shapely.geometry import Point, Polygon
from scipy.stats import gaussian_kde

pd.options.mode.chained_assignment = None


def create_gridpoints(bbox, resolution, return_coords=False):
    assert resolution > 0, \
        "Invalid resolution."
    assert isinstance(bbox, gpd.GeoDataFrame), \
        'bbox must be geopandas GeoDataFrame.'
    bounds = bbox.bounds
    b_s, b_w = bounds.min().values[1], bounds.min().values[0]
    b_n, b_e = bounds.max().values[3], bounds.max().values[2]
    nlon = int(np.ceil((b_e-b_w) / (resolution/111.32)))
    nlat = int(np.ceil((b_n-b_s) / (resolution/110.57)))
    lonv, latv = np.meshgrid(np.linspace(b_w, b_e, nlon), np.linspace(b_s, b_n, nlat))
    gridpoints = pd.DataFrame(np.vstack([lonv.ravel(), latv.ravel()]).T,
                              columns=['lon', 'lat'])
    gridpoints['geometry'] = gridpoints.apply(lambda x: Point([x['lon'], x['lat']]),
                                              axis=1)
    gridpoints = gpd.GeoDataFrame(gridpoints)
    gridpoints.crs = bbox.crs
    grid_ix = gpd.sjoin(gridpoints, bbox, op='intersects').index.unique()
    if len(grid_ix) == 0:
        raise Exception("resolution too big/coarse. No cells were generated.")
    # elif len(grid_ix) / bbox.area.sum() > 10:
    #     warnings.warn('resolution too fine/small. As consequence, your program' \
    #         + 'may run very slowly.')
    gridpoints = gridpoints.loc[grid_ix]
    gridpoints.index.name = 'places'
    if return_coords:
        return gridpoints, lonv, latv
    return gridpoints


def create_hexagon(l, x, y):
    c = [[x + math.cos(math.radians(angle)) * l, y + math.sin(math.radians(angle)) * l] for angle in range(0, 360, 60)]
    return Polygon(c)


def create_gridhexagonal(bbox, resolution):
    assert resolution > 0, \
        "Invalid resolution."
    assert isinstance(bbox, gpd.GeoDataFrame), \
        'bbox must be geopandas GeoDataFrame.'
    bbox_ = list(bbox.bounds.min().values[:2]) + list(bbox.bounds.max().values[-2:])
    x_min = min(bbox_[0], bbox_[2])
    x_max = max(bbox_[0], bbox_[2])
    y_min = min(bbox_[1], bbox_[3])
    y_max = max(bbox_[1], bbox_[3])
    grid = []
    resolution = resolution/110
    v_step = math.sqrt(3) * resolution
    h_step = 1.5 * resolution
    h_skip = math.ceil(x_min / h_step) - 1
    h_start = h_skip * h_step
    v_skip = math.ceil(y_min / v_step) - 1
    v_start = v_skip * v_step
    h_end = x_max + h_step
    v_end = y_max + v_step
    if v_start - (v_step / 2.0) < y_min:
        v_start_array = [v_start + (v_step / 2.0), v_start]
    else:
        v_start_array = [v_start - (v_step / 2.0), v_start]
    v_start_idx = int(abs(h_skip) % 2)
    c_x = h_start
    c_y = v_start_array[v_start_idx]
    v_start_idx = (v_start_idx + 1) % 2
    while c_x < h_end:
        while c_y < v_end:
            grid.append(create_hexagon(resolution, c_x, c_y))
            c_y += v_step
        c_x += h_step
        c_y = v_start_array[v_start_idx]
        v_start_idx = (v_start_idx + 1) % 2
    grid = gpd.GeoDataFrame(geometry=grid).reset_index()
    grid = grid.rename(columns={'index':'places'}).set_index('places')
    grid.crs = bbox.crs
    if isinstance(bbox, gpd.GeoDataFrame):
        grid = gpd.sjoin(grid, bbox, op='intersects')[grid.columns].drop_duplicates()
    grid['lon'] = grid['geometry'].centroid.x
    grid['lat'] = grid['geometry'].centroid.y
    return grid


class SpatioTemporalMapping(ABC, TransformerMixin, BaseEstimator): # y

    def __init__(self, tfreq, grid, start_time=False, end_time=False):
        assert tfreq.upper() in ['M', 'W', 'D'], \
            "Invalid tfreq. Please choose (m)onthly, (w)eekly or (d)aily."
        assert all([x in grid.columns for x in ['geometry', 'lon', 'lat']]), \
            "Input grid must have `geometry`, `lon` and `lat` columns."
        self._tfreq = tfreq.upper()
        self._grid = grid
        self._start_time = pd.to_datetime(start_time) if start_time else False
        self._end_time = pd.to_datetime(end_time) if end_time else False

    @abstractmethod
    def fit_grid(self, data_points=None):
        pass

    def get_time_data_chunks(self, data_points):
        chunks = data_points.set_index('t').resample(self._tfreq)
        chunks = pd.DataFrame(chunks,
                              columns=['t', 'crime_chunks'])
        chunks = chunks.set_index('t').sort_values('t')
        return chunks.apply(lambda x: x[0], axis=1)

    def get_times_no_data(self, chunks):
        if not self._start_time:
            self._start_time = chunks.index.min()
        if not self._end_time:
            self._end_time = chunks.index.max()
        times_between = pd.date_range(self._start_time, self._end_time, freq='M')
        no_data = {cell:0 for cell in self._grid.index}
        times_no_data = set(times_between) - set(chunks.index)
        if len(times_no_data) == 0:
            return chunks
        times_no_data = pd.DataFrame(index=times_no_data)
        times_no_data['crime_density'] = [no_data] * len(times_no_data)
        return times_no_data

    def fit(self, x, y=None):
        return self

    def transform(self, data_points):
        chunks = self.get_time_data_chunks(data_points)
        stseries = chunks.apply(self.fit_grid).to_frame('crime_density')
        times_no_data = self.get_times_no_data(chunks)
        if len(times_no_data) != len(chunks):
            stseries = stseries.append(times_no_data)
        time_ix = stseries.index
        stseries = pd.json_normalize(data=stseries['crime_density'])
        stseries.index = time_ix
        stseries = stseries.unstack()
        stseries.index.names = ['places','t']
        stseries = stseries.swaplevel().sort_index()
        return stseries


class KDE(SpatioTemporalMapping): # y

    def __init__(self, tfreq, grid, start_time=False, end_time=False, bandwidth='silverman'):
        super().__init__(tfreq, grid, start_time, end_time)
        self._bandwidth = bandwidth
        self._kernel = None

    def fit_grid(self, data_points, as_df=False):
        if len(data_points) < 3:
            crime_density = pd.DataFrame([0]*len(self._grid.index),
                                         index=self._grid.index,
                                         columns=['crime_density'])
        else:
            if isinstance(self._bandwidth, str):
                self._kernel = gaussian_kde(np.vstack([data_points.centroid.x,
                                                    data_points.centroid.y]),
                                            bw_method='silverman')
                self._bandwidth = self._kernel.factor
            else:
                self._kernel = gaussian_kde(np.vstack([data_points.centroid.x,
                                                       data_points.centroid.y]),
                                            bw_method=self._bandwidth)
            crime_density = pd.DataFrame(self._kernel(self._grid[['lon', 'lat']].values.T),
                                         index=self._grid.index, columns=['crime_density'])
        if as_df:
            return crime_density
        return crime_density.to_dict()['crime_density']