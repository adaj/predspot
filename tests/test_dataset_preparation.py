import geopandas as gpd
import pandas as pd
import pytest

from predspot.dataset_preparation import Dataset


def test_dataset_builds_points_in_wgs84(crimes, study_area):
    ds = Dataset(crimes, study_area)
    assert isinstance(ds.crimes, gpd.GeoDataFrame)
    assert ds.crimes.crs.to_epsg() == 4326
    assert pd.api.types.is_datetime64_any_dtype(ds.crimes['t'])
    assert ds.crimes.geometry.geom_type.eq('Point').all()
    assert ds.shape == {'crimes': (len(crimes), 5), 'study_area': (1, 2)}
    assert 'predspot.Dataset' in repr(ds)


def test_dataset_does_not_mutate_input(crimes, study_area):
    before = crimes.copy()
    Dataset(crimes, study_area)
    pd.testing.assert_frame_equal(crimes, before)
    assert 'geometry' not in crimes.columns


def test_dataset_validation(crimes, study_area):
    with pytest.raises(TypeError):
        Dataset(crimes, study_area.geometry.iloc[0])
    with pytest.raises(ValueError, match='missing'):
        Dataset(crimes.drop(columns=['lat']), study_area)
    with pytest.raises(ValueError, match='CRS'):
        Dataset(crimes, study_area.set_crs(None, allow_override=True))


def test_train_test_split(dataset):
    train, test = dataset.train_test_split(test_size=0.2, random_state=0)
    assert len(train.crimes) + len(test.crimes) == len(dataset.crimes)
    assert set(train.crimes.index).isdisjoint(test.crimes.index)
    with pytest.raises(ValueError):
        dataset.train_test_split(test_size=1.5)


def test_plot(dataset):
    import matplotlib
    matplotlib.use('Agg')
    ax = dataset.plot(crime_samples=50)
    assert ax is not None
