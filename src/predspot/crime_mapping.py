"""
Crime Mapping Module
====================

Spatial and temporal crime mapping. This module turns a set of georeferenced,
timestamped crime events into a *spatio-temporal series*: a value per grid cell
per time period. Two families of mapping are available:

* [`KDE`][predspot.crime_mapping.KDE] — kernel density estimation evaluated on a grid of **points**
  (see [`create_gridpoints`][predspot.crime_mapping.create_gridpoints]). This is the default
  approach of Predspot.
* [`QuadratCount`][predspot.crime_mapping.QuadratCount] — plain event counts per **cell** of a
polygonal grid
  (see [`create_gridhexagonal`][predspot.crime_mapping.create_gridhexagonal] and
  [`create_gridsquares`][predspot.crime_mapping.create_gridsquares]).

Both produce the same output format, a ``pandas.Series`` named
``crime_density`` indexed by ``(t, places)``, so they are interchangeable
inside [`PredictionPipeline`][predspot.ml_modelling.PredictionPipeline].

The study area itself can be fetched from OpenStreetMap with
[`load_study_area`][predspot.crime_mapping.load_study_area] (requires the optional ``osmnx``
dependency).
"""

__author__ = "Adelson Araujo"

import logging
import math
from abc import ABC, abstractmethod

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from shapely.geometry import Polygon
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

WGS84 = "EPSG:4326"

# Approximate length of one degree at the equator, in km. Used to translate a
# resolution given in km into degrees when building grids in WGS84.
KM_PER_DEG_LON = 111.32
KM_PER_DEG_LAT = 110.57

# Public aliases accepted for the time frequency and the pandas offset alias
# they map to. Old pandas used 'M' for month end; pandas >= 2.2 uses 'ME'.
TFREQ_ALIASES = {
    "M": "ME",
    "ME": "ME",
    "MONTH": "ME",
    "MONTHLY": "ME",
    "W": "W",
    "WEEK": "W",
    "WEEKLY": "W",
    "D": "D",
    "DAY": "D",
    "DAILY": "D",
}


def normalize_tfreq(tfreq):
    """Translate a user-facing time frequency into a pandas offset alias.

    Args:
        tfreq (str): One of ``'M'``/``'ME'`` (monthly), ``'W'`` (weekly) or
            ``'D'`` (daily), case-insensitive.

    Returns:
        str: The pandas offset alias (``'ME'``, ``'W'`` or ``'D'``).

    Raises:
        ValueError: If ``tfreq`` is not one of the accepted values.
    """
    key = str(tfreq).upper()
    if key not in TFREQ_ALIASES:
        raise ValueError(f"Invalid tfreq {tfreq!r}. Choose (M)onthly, (W)eekly or (D)aily.")
    return TFREQ_ALIASES[key]


def tfreq_offset(tfreq):
    """Return the ``pandas.DateOffset`` that advances one period of ``tfreq``."""
    alias = normalize_tfreq(tfreq)
    if alias == "ME":
        return pd.offsets.MonthEnd(1)
    if alias == "W":
        return pd.offsets.Week(1)
    return pd.offsets.Day(1)


def load_study_area(place, crs=WGS84, which_result=None):
    """
    Fetch the boundary polygon of a place from OpenStreetMap.

    Uses `osmnx <https://osmnx.readthedocs.io>`_ (optional dependency:
    ``pip install predspot[osm]``) to geocode ``place`` with Nominatim and
    return its administrative boundary, ready to be used as the
    ``study_area`` of [`Dataset`][predspot.Dataset] or as the ``bbox`` of the
    ``create_grid*`` functions.

    Args:
        place (str or list): Name of the place as Nominatim understands it,
            e.g. ``"Natal, Rio Grande do Norte, Brazil"``. A list of names
            returns one row per place.
        crs (str or pyproj.CRS): CRS of the returned GeoDataFrame (default WGS84).
        which_result (int, optional): Forwarded to
            ``osmnx.geocode_to_gdf`` to pick a specific Nominatim match when
            the first one is not the boundary you want.

    Returns:
        GeoDataFrame: One row per place with ``name``, ``display_name``,
        ``osm_type``, ``osm_id`` and a (Multi)Polygon ``geometry``.

    Raises:
        ImportError: If ``osmnx`` is not installed.
        ValueError: If Nominatim returns a point instead of a boundary
            polygon for the query.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "load_study_area requires the optional dependency `osmnx`: pip install predspot[osm]"
        ) from exc
    logger.debug("Geocoding study area %r with osmnx", place)
    gdf = ox.geocode_to_gdf(place, which_result=which_result)
    if not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        bad = gdf.loc[~gdf.geom_type.isin(["Polygon", "MultiPolygon"]), "display_name"]
        raise ValueError(
            "OpenStreetMap returned a non-polygon geometry for "
            f"{bad.tolist()}. Try a more specific query (e.g. add the state "
            "and country) or a different `which_result`."
        )
    columns = [c for c in ("name", "display_name", "osm_type", "osm_id") if c in gdf.columns]
    gdf = gdf[columns + ["geometry"]].reset_index(drop=True)
    return gdf.to_crs(crs)


def get_city_shape(place_query):
    """
    Fetch the shape (polygon) of a city or region from OpenStreetMap.

    A thin wrapper around ``osmnx.geocode_to_gdf`` that returns the raw
    Nominatim result. Prefer [`load_study_area`][predspot.crime_mapping.load_study_area]
    when you want the result validated (polygon geometry, tidy columns).
    Requires the optional ``osmnx`` dependency (``pip install predspot[osm]``).

    Args:
        place_query (str): Name of the place, in a format Nominatim accepts,
            e.g. ``"Natal, RN, Brazil"`` or ``"Rio Grande do Norte, Brazil"``.

    Returns:
        GeoDataFrame: One row (or more, if the query is ambiguous) with the
        place geometry (Polygon/MultiPolygon) in WGS84.

    Example:
        >>> city = get_city_shape("Natal, RN, Brazil")
        >>> city.geometry.iloc[0]  # shapely Polygon/MultiPolygon
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "get_city_shape requires the optional dependency `osmnx`: pip install predspot[osm]"
        ) from exc
    return ox.geocode_to_gdf(place_query)


def _check_bbox(bbox):
    if not isinstance(bbox, gpd.GeoDataFrame):
        raise TypeError("bbox must be a geopandas GeoDataFrame.")
    if bbox.crs is None:
        raise ValueError('bbox must have a CRS (e.g. bbox.set_crs("EPSG:4326")).')
    if len(bbox) == 0:
        raise ValueError("bbox is empty.")


def _wgs84_bounds(bbox):
    """Total bounds (west, south, east, north) of ``bbox`` in WGS84 degrees."""
    return bbox.to_crs(WGS84).total_bounds


def _clip_to_bbox(grid, bbox):
    """Keep only grid rows that intersect ``bbox`` (both in the same CRS)."""
    keep = gpd.sjoin(grid, bbox[["geometry"]], how="inner", predicate="intersects").index.unique()
    if len(keep) == 0:
        raise ValueError("resolution too big/coarse. No cells intersect the study area.")
    return grid.loc[grid.index.isin(keep)]


def _add_centroid_lonlat(grid):
    """Add ``lon``/``lat`` columns with cell centroids computed in a projected CRS."""
    projected = grid.geometry.to_crs(grid.estimate_utm_crs())
    centroids = projected.centroid.to_crs(WGS84)
    grid = grid.copy()
    grid["lon"] = centroids.x.values
    grid["lat"] = centroids.y.values
    return grid


def create_gridpoints(bbox, resolution, return_coords=False):
    """
    Create a regular grid of points covering a study area.

    This is the grid used by [`KDE`][predspot.crime_mapping.KDE]: the density is evaluated at each
    point. The grid is built in WGS84 with the requested spacing and then
    re-projected to the CRS of ``bbox``.

    Args:
        bbox (GeoDataFrame): Study area (any CRS, must be set).
        resolution (float): Spacing between points, in kilometers.
        return_coords (bool): If True, also return the full ``lon``/``lat``
            meshgrids (before clipping), useful for contour plots.

    Returns:
        GeoDataFrame: Points intersecting ``bbox`` with ``lon``, ``lat`` and
        ``geometry`` columns and an index named ``places``. When
        ``return_coords`` is True, a tuple ``(gridpoints, lonv, latv)``.
    """
    if resolution <= 0:
        raise ValueError("resolution must be a positive number of kilometers.")
    _check_bbox(bbox)
    logger.debug("Creating point grid with resolution %s km", resolution)

    b_w, b_s, b_e, b_n = _wgs84_bounds(bbox)
    nlon = max(int(np.ceil((b_e - b_w) / (resolution / KM_PER_DEG_LON))), 2)
    nlat = max(int(np.ceil((b_n - b_s) / (resolution / KM_PER_DEG_LAT))), 2)
    lonv, latv = np.meshgrid(np.linspace(b_w, b_e, nlon), np.linspace(b_s, b_n, nlat))
    lon, lat = lonv.ravel(), latv.ravel()
    gridpoints = gpd.GeoDataFrame(
        {"lon": lon, "lat": lat}, geometry=gpd.points_from_xy(lon, lat), crs=WGS84
    ).to_crs(bbox.crs)
    gridpoints = _clip_to_bbox(gridpoints, bbox)
    gridpoints.index.name = "places"
    if return_coords:
        return gridpoints, lonv, latv
    return gridpoints


def create_hexagon(side, x, y):
    """
    Create a flat-topped hexagonal polygon.

    Args:
        side (float): Length of the hexagon side (circumradius), in the units
            of ``x``/``y``.
        x (float): X-coordinate of the center.
        y (float): Y-coordinate of the center.

    Returns:
        Polygon: The hexagon.
    """
    return Polygon(
        [
            (x + math.cos(math.radians(angle)) * side, y + math.sin(math.radians(angle)) * side)
            for angle in range(0, 360, 60)
        ]
    )


def create_gridhexagonal(bbox, resolution):
    """
    Create a hexagonal grid covering a study area.

    Each hexagon has the same area as a square of side ``resolution`` km, so
    hexagonal and square grids of the same resolution are comparable. The
    grid is built in WGS84 and re-projected to the CRS of ``bbox``.

    Args:
        bbox (GeoDataFrame): Study area (any CRS, must be set).
        resolution (float): Equivalent square side, in kilometers.

    Returns:
        GeoDataFrame: Hexagons intersecting ``bbox`` with ``geometry``,
        ``lon`` and ``lat`` (centroid) columns and an index named ``places``.
    """
    if resolution <= 0:
        raise ValueError("resolution must be a positive number of kilometers.")
    _check_bbox(bbox)
    logger.debug("Creating hexagonal grid with resolution %s km", resolution)

    # Side length such that the hexagon area equals resolution**2.
    side_km = math.sqrt(resolution**2 * 2 / (3 * math.sqrt(3)))
    side = side_km / KM_PER_DEG_LAT  # degrees (isotropic approximation)
    x_min, y_min, x_max, y_max = _wgs84_bounds(bbox)

    v_step = math.sqrt(3) * side
    h_step = 1.5 * side
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

    hexagons = []
    v_start_idx = int(abs(h_skip) % 2)
    c_x = h_start
    while c_x < h_end:
        c_y = v_start_array[v_start_idx]
        while c_y < v_end:
            hexagons.append(create_hexagon(side, c_x, c_y))
            c_y += v_step
        c_x += h_step
        v_start_idx = (v_start_idx + 1) % 2

    grid = gpd.GeoDataFrame(geometry=hexagons, crs=WGS84).to_crs(bbox.crs)
    grid = _clip_to_bbox(grid, bbox)
    grid = _add_centroid_lonlat(grid)
    grid.index.name = "places"
    return grid


def create_gridsquares(bbox, resolution=1):
    """
    Create a grid of square cells covering a study area.

    Args:
        bbox (GeoDataFrame): Study area (any CRS, must be set).
        resolution (float): Side of each square, in kilometers.

    Returns:
        GeoDataFrame: Squares intersecting ``bbox`` with ``geometry``,
        ``lon`` and ``lat`` (centroid) columns and an index named ``places``.
    """
    if resolution <= 0:
        raise ValueError("resolution must be a positive number of kilometers.")
    _check_bbox(bbox)
    logger.debug("Creating square grid with resolution %s km", resolution)

    x0, y0, xf, yf = _wgs84_bounds(bbox)
    dx = resolution / KM_PER_DEG_LON
    dy = resolution / KM_PER_DEG_LAT
    xs = np.arange(x0, xf, dx)
    ys = np.arange(y0, yf, dy)
    squares = [
        Polygon([(x, y), (x + dx, y), (x + dx, y + dy), (x, y + dy)]) for x in xs for y in ys
    ]
    grid = gpd.GeoDataFrame(geometry=squares, crs=WGS84).to_crs(bbox.crs)
    grid = _clip_to_bbox(grid, bbox)
    grid = _add_centroid_lonlat(grid)
    grid.index.name = "places"
    return grid


class SpatioTemporalMapping(ABC, TransformerMixin, BaseEstimator):
    """
    Abstract base class for spatio-temporal crime mapping.

    Subclasses implement ``fit_grid``, which maps the events of a single
    time period onto the grid. ``transform`` takes care of splitting the
    events into periods, filling periods with no events and assembling the
    result into a series indexed by ``(t, places)``.

    Args:
        tfreq (str): Time frequency: ``'M'`` (monthly), ``'W'`` (weekly) or
            ``'D'`` (daily).
        grid (GeoDataFrame): Spatial grid with ``geometry``, ``lon`` and
            ``lat`` columns, as produced by the ``create_grid*`` functions.
        start_time (str or datetime, optional): Force the series to start at
            this time (periods without events are filled with zeros).
        end_time (str or datetime, optional): Force the series to end at this
            time.
    """

    def __init__(self, tfreq, grid, start_time=None, end_time=None):
        self.tfreq = tfreq
        self.grid = grid
        self.start_time = start_time
        self.end_time = end_time

        self._tfreq = normalize_tfreq(tfreq)
        missing = [c for c in ("geometry", "lon", "lat") if c not in grid.columns]
        if missing:
            raise ValueError(
                f"Input grid must have `geometry`, `lon` and `lat` columns; missing {missing}."
            )
        self._grid = grid
        self._start_time = pd.to_datetime(start_time) if start_time else None
        self._end_time = pd.to_datetime(end_time) if end_time else None
        logger.debug(
            "%s initialised with tfreq=%s and %d places",
            type(self).__name__,
            self._tfreq,
            len(grid),
        )

    @abstractmethod
    def fit_grid(self, data_points):
        """
        Map the events of one time period onto the grid.

        Args:
            data_points (GeoDataFrame): Events of a single period.

        Returns:
            dict: ``{place: value}`` for every place of the grid.
        """

    def fit(self, x=None, y=None):
        """No-op; present for scikit-learn compatibility."""
        return self

    def _time_index(self, chunk_labels):
        """Full time index, extended to ``start_time``/``end_time`` if given."""
        start = self._start_time if self._start_time is not None else chunk_labels.min()
        end = self._end_time if self._end_time is not None else chunk_labels.max()
        full = pd.date_range(start, end, freq=self._tfreq)
        return full.union(chunk_labels[(chunk_labels >= start) & (chunk_labels <= end)])

    def transform(self, data_points):
        """
        Build the spatio-temporal series from crime events.

        Args:
            data_points (GeoDataFrame): Events with a ``t`` timestamp column
                and point geometries (e.g. ``Dataset.crimes``).

        Returns:
            pandas.Series: Values named ``crime_density`` indexed by
            ``(t, places)``, sorted.
        """
        if "t" not in data_points.columns:
            raise ValueError("data_points must have a `t` timestamp column.")
        events = data_points.set_index(pd.DatetimeIndex(data_points["t"])).sort_index()
        chunks = {label: chunk for label, chunk in events.resample(self._tfreq)}
        labels = pd.DatetimeIndex(list(chunks.keys()))
        time_index = self._time_index(labels)
        logger.debug("Mapping %d events over %d periods", len(events), len(time_index))

        zeros = dict.fromkeys(self._grid.index, 0.0)
        rows = []
        for t in time_index:
            chunk = chunks.get(t)
            if chunk is None or len(chunk) == 0:
                rows.append(zeros)
            else:
                rows.append(self.fit_grid(chunk))
        frame = pd.DataFrame(rows, index=time_index)
        frame = frame.reindex(columns=self._grid.index)
        stseries = frame.stack()  # noqa: PD013 - long format with (t, places) index
        stseries.index.names = ["t", "places"]
        stseries.name = "crime_density"
        return stseries.sort_index()


class KDE(SpatioTemporalMapping):
    """
    Kernel density estimation of crime events evaluated on grid points.

    For every time period a Gaussian KDE is fitted to the event coordinates
    and evaluated at the grid points (``lon``/``lat`` columns of the grid).
    Periods with fewer than 3 events get a density of zero everywhere.

    Args:
        tfreq (str): Time frequency (``'M'``, ``'W'`` or ``'D'``).
        grid (GeoDataFrame): Point grid, see
            [`create_gridpoints`][predspot.crime_mapping.create_gridpoints].
        start_time (str or datetime, optional): See
            [`SpatioTemporalMapping`][predspot.crime_mapping.SpatioTemporalMapping].
        end_time (str or datetime, optional): See
            [`SpatioTemporalMapping`][predspot.crime_mapping.SpatioTemporalMapping].
        bandwidth (str or float): ``'silverman'`` (default) or ``'scott'`` to
            estimate the bandwidth from the first period with enough events
            and keep it fixed afterwards (so densities are comparable across
            time), or a positive number used directly as the KDE factor.
    """

    def __init__(self, tfreq, grid, start_time=None, end_time=None, bandwidth="silverman"):
        super().__init__(tfreq, grid, start_time, end_time)
        self.bandwidth = bandwidth
        if isinstance(bandwidth, str):
            method = bandwidth.lower()
            if method == "auto":
                method = "silverman"
            if method not in ("silverman", "scott"):
                raise ValueError("bandwidth must be 'silverman', 'scott' or a number.")
            self._bw_method = method
            self._factor = None
        else:
            if bandwidth <= 0:
                raise ValueError("bandwidth must be a positive number.")
            self._bw_method = None
            self._factor = float(bandwidth)
        self._kernel = None

    @property
    def factor(self):
        """float or None: KDE factor in use (``None`` until the first fit)."""
        return self._factor

    def fit_grid(self, data_points, as_df=False):
        """
        Fit the KDE to the events of one period and evaluate it on the grid.

        Args:
            data_points (GeoDataFrame): Events of a single period.
            as_df (bool): If True return a DataFrame instead of a dict.

        Returns:
            dict or DataFrame: Density at each grid point.
        """
        if len(data_points) < 3:
            values = np.zeros(len(self._grid))
        else:
            xy = np.vstack([data_points.geometry.x.values, data_points.geometry.y.values])
            bw = self._factor if self._factor is not None else self._bw_method
            self._kernel = gaussian_kde(xy, bw_method=bw)
            if self._factor is None:
                self._factor = float(self._kernel.factor)
                logger.debug(
                    "KDE bandwidth factor estimated with %s: %.5f", self._bw_method, self._factor
                )
            values = self._kernel(self._grid[["lon", "lat"]].values.T)
        density = pd.DataFrame({"crime_density": values}, index=self._grid.index)
        if as_df:
            return density
        return density["crime_density"].to_dict()


class QuadratCount(SpatioTemporalMapping):
    """
    Count of crime events per grid cell (quadrat count).

    An alternative to [`KDE`][predspot.crime_mapping.KDE] that works on polygonal grids (hexagons or
    squares, see [`create_gridhexagonal`][predspot.crime_mapping.create_gridhexagonal] and
    [`create_gridsquares`][predspot.crime_mapping.create_gridsquares]):
    the value of a cell in a period is the number of events that fall in it.

    Args:
        tfreq (str): Time frequency (``'M'``, ``'W'`` or ``'D'``).
        grid (GeoDataFrame): Polygonal grid.
        start_time (str or datetime, optional): See
            [`SpatioTemporalMapping`][predspot.crime_mapping.SpatioTemporalMapping].
        end_time (str or datetime, optional): See
            [`SpatioTemporalMapping`][predspot.crime_mapping.SpatioTemporalMapping].
    """

    def __init__(self, tfreq, grid, start_time=None, end_time=None):
        super().__init__(tfreq, grid, start_time, end_time)
        if not grid.geom_type.isin(["Polygon", "MultiPolygon"]).all():
            raise ValueError(
                "QuadratCount requires a polygonal grid "
                "(see create_gridhexagonal / create_gridsquares)."
            )

    def fit_grid(self, data_points, as_df=False):
        """
        Count the events of one period in each cell of the grid.

        Args:
            data_points (GeoDataFrame): Events of a single period.
            as_df (bool): If True return a DataFrame instead of a dict.

        Returns:
            dict or DataFrame: Number of events per cell.
        """
        points = data_points[["geometry"]].to_crs(self._grid.crs)
        joined = gpd.sjoin(points, self._grid[["geometry"]], how="inner", predicate="within")
        # The right index column is named after the grid index ('places');
        # fall back to geopandas' default name otherwise.
        col = "places" if "places" in joined.columns else "index_right"
        counts = joined.groupby(col).size().reindex(self._grid.index, fill_value=0)
        density = pd.DataFrame(
            {"crime_density": counts.astype(float).values}, index=self._grid.index
        )
        if as_df:
            return density
        return density["crime_density"].to_dict()
