import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from predspot.dataset_preparation import Dataset

# ~10 x 10 km box in WGS84 (west, south, east, north)
BOUNDS = (-35.30, -5.90, -35.20, -5.80)


@pytest.fixture(scope='session')
def study_area():
    return gpd.GeoDataFrame({'name': ['test']},
                            geometry=[box(*BOUNDS)], crs='EPSG:4326')


@pytest.fixture(scope='session')
def crimes():
    """2 years of events: a uniform background plus one Gaussian hotspot."""
    rng = np.random.default_rng(42)
    n_bg, n_hot = 1500, 1500
    west, south, east, north = BOUNDS
    lon = np.concatenate([rng.uniform(west, east, n_bg), rng.normal(-35.23, 0.01, n_hot)])
    lat = np.concatenate([rng.uniform(south, north, n_bg), rng.normal(-5.83, 0.01, n_hot)])
    start = pd.Timestamp('2019-01-01')
    seconds = rng.integers(0, 730 * 24 * 3600, n_bg + n_hot)
    return pd.DataFrame({
        'tag': rng.choice(['burglary', 'assault'], n_bg + n_hot, p=[0.8, 0.2]),
        't': start + pd.to_timedelta(seconds, unit='s'),
        'lon': np.clip(lon, west, east), 'lat': np.clip(lat, south, north),
    })


@pytest.fixture(scope='session')
def dataset(crimes, study_area):
    return Dataset(crimes, study_area)
