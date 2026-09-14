import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from predspot import crime_mapping as cm


@pytest.mark.parametrize(
    "tfreq,expected",
    [("M", "ME"), ("m", "ME"), ("ME", "ME"), ("W", "W"), ("D", "D"), ("daily", "D")],
)
def test_normalize_tfreq(tfreq, expected):
    assert cm.normalize_tfreq(tfreq) == expected


def test_normalize_tfreq_invalid():
    with pytest.raises(ValueError):
        cm.normalize_tfreq("Y")


def test_create_gridpoints(study_area):
    grid = cm.create_gridpoints(study_area, resolution=1)
    assert grid.index.name == "places"
    assert {"lon", "lat", "geometry"} <= set(grid.columns)
    # ~11 x 11 points for a ~10x10 km box at 1 km spacing
    assert 100 <= len(grid) <= 200
    assert grid.geometry.geom_type.eq("Point").all()
    assert grid.crs == study_area.crs


def test_create_gridpoints_return_coords(study_area):
    grid, lonv, latv = cm.create_gridpoints(study_area, resolution=2, return_coords=True)
    assert lonv.shape == latv.shape
    assert len(grid) <= lonv.size


def test_create_gridpoints_projected_bbox(study_area):
    projected = study_area.to_crs(study_area.estimate_utm_crs())
    grid = cm.create_gridpoints(projected, resolution=1)
    assert grid.crs == projected.crs
    assert 100 <= len(grid) <= 200


def test_create_gridpoints_errors(study_area):
    with pytest.raises(ValueError):
        cm.create_gridpoints(study_area, resolution=0)
    with pytest.raises(TypeError):
        cm.create_gridpoints(study_area.geometry, resolution=1)
    # a diamond: the corners of its bounding box fall outside the polygon
    import geopandas as gpd
    from shapely.geometry import Polygon

    diamond = Polygon([(-35.25, -5.90), (-35.20, -5.85), (-35.25, -5.80), (-35.30, -5.85)])
    sliver = gpd.GeoDataFrame(geometry=[diamond], crs="EPSG:4326")
    with pytest.raises(ValueError, match="coarse"):
        cm.create_gridpoints(sliver, resolution=10000)


def test_create_gridhexagonal(study_area):
    grid = cm.create_gridhexagonal(study_area, resolution=1)
    assert grid.index.name == "places"
    assert grid.geometry.geom_type.eq("Polygon").all()
    assert {"lon", "lat"} <= set(grid.columns)
    # every centroid must lie inside its own hexagon
    centroids = gpd.points_from_xy(grid.lon, grid.lat)
    assert all(geom.contains(pt) for geom, pt in zip(grid.geometry, centroids, strict=True))
    # equal-area hexagons: the study area (~100 km2) needs ~100-150 cells of 1 km2
    assert 90 <= len(grid) <= 170
    assert grid.geometry.union_all().covers(study_area.geometry.iloc[0])


def test_create_gridsquares(study_area):
    grid = cm.create_gridsquares(study_area, resolution=1)
    assert grid.index.name == "places"
    assert grid.geometry.geom_type.eq("Polygon").all()
    assert 100 <= len(grid) <= 150
    assert grid.geometry.union_all().covers(study_area.geometry.iloc[0])


@pytest.fixture(scope="module")
def points_grid(study_area):
    return cm.create_gridpoints(study_area, resolution=1)


@pytest.fixture(scope="module")
def hex_grid(study_area):
    return cm.create_gridhexagonal(study_area, resolution=1)


def test_kde_transform(dataset, points_grid):
    kde = cm.KDE(tfreq="M", grid=points_grid)
    st = kde.fit_transform(dataset.crimes)
    assert isinstance(st, pd.Series)
    assert st.name == "crime_density"
    assert st.index.names == ["t", "places"]
    times = st.index.get_level_values("t").unique()
    assert len(times) == 24  # 2019-01 .. 2020-12
    assert (times == pd.date_range("2019-01-31", "2020-12-31", freq="ME")).all()
    assert len(st) == 24 * len(points_grid)
    assert not st.isna().any()
    assert (st >= 0).all()
    assert kde.factor is not None and kde.factor > 0
    # the hotspot around (-35.23, -5.83) must be denser than the far corner
    month = st.xs(times[5], level="t")
    hot = ((points_grid.lon - (-35.23)).abs() < 0.005) & ((points_grid.lat - (-5.83)).abs() < 0.005)
    cold = (points_grid.lon < -35.29) & (points_grid.lat < -5.89)
    assert month[hot.values].mean() > month[cold.values].mean()


def test_kde_bandwidth_options(dataset, points_grid):
    fixed = cm.KDE(tfreq="M", grid=points_grid, bandwidth=0.3)
    fixed.fit_transform(dataset.crimes)
    assert fixed.factor == 0.3
    scott = cm.KDE(tfreq="M", grid=points_grid, bandwidth="scott")
    scott.fit_transform(dataset.crimes)
    assert scott.factor > 0
    auto = cm.KDE(tfreq="M", grid=points_grid, bandwidth="auto")
    assert auto._bw_method == "silverman"
    with pytest.raises(ValueError):
        cm.KDE(tfreq="M", grid=points_grid, bandwidth="gaussian")
    with pytest.raises(ValueError):
        cm.KDE(tfreq="M", grid=points_grid, bandwidth=-1)


def test_kde_start_end_time(dataset, points_grid):
    kde = cm.KDE(tfreq="M", grid=points_grid, start_time="2018-06-01", end_time="2021-03-31")
    st = kde.fit_transform(dataset.crimes)
    times = st.index.get_level_values("t").unique()
    assert times.min() == pd.Timestamp("2018-06-30")
    assert times.max() == pd.Timestamp("2021-03-31")
    assert (st.xs(pd.Timestamp("2018-06-30"), level="t") == 0).all()


def test_kde_weekly_and_daily(dataset, points_grid):
    weekly = cm.KDE(tfreq="W", grid=points_grid).fit_transform(dataset.crimes)
    assert 100 <= len(weekly.index.get_level_values("t").unique()) <= 106
    small = dataset.crimes[dataset.crimes["t"] < "2019-02-01"]
    daily = cm.KDE(tfreq="D", grid=points_grid).fit_transform(small)
    assert len(daily.index.get_level_values("t").unique()) == 31


def test_kde_few_points_gives_zeros(dataset, points_grid):
    two = dataset.crimes.iloc[:2]
    st = cm.KDE(tfreq="M", grid=points_grid).fit_transform(two)
    assert (st == 0).all()


def test_kde_grid_validation(points_grid):
    with pytest.raises(ValueError, match="lon"):
        cm.KDE(tfreq="M", grid=points_grid.drop(columns=["lon"]))


def test_quadrat_count(dataset, hex_grid):
    qc = cm.QuadratCount(tfreq="M", grid=hex_grid)
    st = qc.fit_transform(dataset.crimes)
    assert st.name == "crime_density"
    assert st.index.names == ["t", "places"]
    assert len(st.index.get_level_values("t").unique()) == 24
    assert len(st) == 24 * len(hex_grid)
    assert (st >= 0).all()
    assert np.allclose(st, np.round(st))
    # every event falls in exactly one hexagon, so counts add up
    assert st.sum() == len(dataset.crimes)


def test_quadrat_count_squares(dataset, study_area):
    grid = cm.create_gridsquares(study_area, resolution=2)
    st = cm.QuadratCount(tfreq="W", grid=grid).fit_transform(dataset.crimes)
    assert st.sum() == len(dataset.crimes)


def test_quadrat_count_requires_polygons(points_grid):
    with pytest.raises(ValueError, match="polygonal"):
        cm.QuadratCount(tfreq="M", grid=points_grid)
