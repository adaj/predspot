import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import Polygon

from predspot import Dataset, synthetic
from predspot.synthetic import generate_crimes


@pytest.fixture(scope="module")
def irregular_area():
    poly = Polygon(
        [(-35.30, -5.90), (-35.18, -5.92), (-35.16, -5.85), (-35.20, -5.82),
         (-35.19, -5.78), (-35.28, -5.77), (-35.32, -5.84)]
    )  # fmt: skip
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326")


def test_generate_crimes_basic(irregular_area):
    crimes, hotspots = generate_crimes(
        irregular_area, n_events=3000, n_hotspots=3, seed=0, return_hotspots=True
    )
    assert list(crimes.columns) == ["tag", "t", "lon", "lat"]
    assert len(crimes) == 3000
    assert crimes["t"].is_monotonic_increasing
    assert crimes["t"].between("2019-01-01", "2020-12-31").all()
    polygon = irregular_area.geometry.iloc[0]
    assert shapely.contains_xy(polygon, crimes["lon"].values, crimes["lat"].values).all()
    assert len(hotspots) == 3
    assert hotspots.geometry.within(polygon).all()
    assert np.isclose(hotspots["share"].sum(), 0.7)
    # it plugs directly into Dataset
    assert Dataset(crimes, irregular_area).shape["crimes"][0] == 3000


def test_generate_crimes_is_reproducible(study_area):
    a = generate_crimes(study_area, n_events=500, seed=123)
    b = generate_crimes(study_area, n_events=500, seed=123)
    pd.testing.assert_frame_equal(a, b)
    c = generate_crimes(study_area, n_events=500, seed=124)
    assert not a.equals(c)


def test_hotspots_concentrate_events(study_area):
    crimes, hotspots = generate_crimes(
        study_area, n_events=4000, n_hotspots=1, hotspot_share=0.8, hotspot_sd_km=0.3,
        seed=1, return_hotspots=True,
    )  # fmt: skip
    center = hotspots.geometry.iloc[0]
    d_lon = (crimes["lon"] - center.x) * 111.32 * np.cos(np.radians(center.y))
    d_lat = (crimes["lat"] - center.y) * 110.57
    within_1km = np.hypot(d_lon, d_lat) < 1.0
    # ~80% hotspot events within ~3 sd, plus a little background in a 100 km2 box
    assert 0.7 < within_1km.mean() < 0.9


def test_uniform_when_no_hotspots(study_area):
    crimes = generate_crimes(study_area, n_events=4000, n_hotspots=0, seed=2)
    west, south, east, north = study_area.total_bounds
    # split the box into 4 quadrants; each should hold ~25% of the events
    q = ((crimes["lon"] > (west + east) / 2).astype(int) * 2
         + (crimes["lat"] > (south + north) / 2).astype(int))  # fmt: skip
    shares = q.value_counts(normalize=True)
    assert len(shares) == 4 and (shares.between(0.2, 0.3)).all()


def test_temporal_patterns(study_area):
    crimes = generate_crimes(
        study_area, n_events=20000, trend=1.0, annual_amplitude=0.5, annual_peak_month=7,
        weekly_profile=(1, 1, 1, 1, 1, 1, 6), hourly_profile=None, seed=3,
    )  # fmt: skip
    t = crimes["t"]
    # weekly: Sunday gets 6/12 = 50% of the events
    assert 0.45 < (t.dt.dayofweek == 6).mean() < 0.55
    # no hourly profile: hours roughly uniform
    hours = t.dt.hour.value_counts(normalize=True)
    assert hours.max() < 0.07
    # trend: the last year has more events than the first year
    per_year = t.dt.year.value_counts()
    assert per_year[2020] > 1.3 * per_year[2019]
    # annual cycle peaks in July
    assert t.dt.month.value_counts().idxmax() in (6, 7, 8)


def test_tags_weights(study_area):
    crimes = generate_crimes(study_area, n_events=5000, tags={"theft": 3, "fraud": 1}, seed=4)
    shares = crimes["tag"].value_counts(normalize=True)
    assert set(shares.index) == {"theft", "fraud"}
    assert 0.7 < shares["theft"] < 0.8
    listed = generate_crimes(study_area, n_events=200, tags=["a", "b", "c"], seed=4)
    assert set(listed["tag"]) == {"a", "b", "c"}


def test_projected_study_area(study_area):
    projected = study_area.to_crs(study_area.estimate_utm_crs())
    crimes = generate_crimes(projected, n_events=300, seed=5)
    west, south, east, north = study_area.total_bounds
    assert crimes["lon"].between(west, east).all()
    assert crimes["lat"].between(south, north).all()


def test_validation(study_area):
    with pytest.raises(ValueError):
        generate_crimes(study_area, n_events=0)
    with pytest.raises(ValueError):
        generate_crimes(study_area, n_events=10, hotspot_share=1.5)
    with pytest.raises(ValueError):
        generate_crimes(study_area, n_events=10, tags={})
    with pytest.raises(ValueError, match="7 values"):
        generate_crimes(study_area, n_events=10, weekly_profile=(1, 2))
    with pytest.raises(ValueError, match="24 values"):
        generate_crimes(study_area, n_events=10, hourly_profile=(1, 2))
    with pytest.raises(ValueError, match="end must be after"):
        generate_crimes(study_area, n_events=10, start="2020-01-01", end="2019-01-01")


def test_temporal_intensity_shape():
    start, end = pd.Timestamp("2019-01-01"), pd.Timestamp("2019-12-31")
    stamps = pd.date_range(start, end, freq="D")
    lam = synthetic.temporal_intensity(stamps, start, end, trend=1.0)
    assert lam.shape == (len(stamps),)
    assert np.isclose(lam[0], 1.0) and np.isclose(lam[-1], 2.0)
