import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from predspot import crime_mapping as cm
from predspot import feature_engineering as fe
from predspot.utilities import PandasFeatureUnion


@pytest.fixture(scope='module')
def stseries(dataset, study_area):
    grid = cm.create_gridpoints(study_area, resolution=2)
    return cm.KDE(tfreq='M', grid=grid).fit_transform(dataset.crimes)


def _check_features(X, stseries, label, lags, first_time):
    assert list(X.columns) == [f'{label}_{i}' for i in range(1, lags + 1)]
    assert X.index.names == ['t', 'places']
    times = X.index.get_level_values('t').unique()
    last_obs = stseries.index.get_level_values('t').max()
    assert times.max() == last_obs + pd.offsets.MonthEnd(1)  # next period row
    assert times.min() == first_time
    n_places = stseries.index.get_level_values('places').nunique()
    assert len(X) == len(times) * n_places
    assert not X.isna().any().any()


def test_ar_features(stseries):
    X = fe.AR(lags=3, tfreq='M').fit_transform(stseries)
    _check_features(X, stseries, 'ar', 3, pd.Timestamp('2019-04-30'))
    # ar_1 at t equals the series at t-1
    place = stseries.index.get_level_values('places')[0]
    assert np.isclose(X.loc[(pd.Timestamp('2019-04-30'), place), 'ar_1'],
                      stseries.loc[(pd.Timestamp('2019-03-31'), place)])


def test_diff_features(stseries):
    X = fe.Diff(lags=2, tfreq='M').fit_transform(stseries)
    _check_features(X, stseries, 'diff', 2, pd.Timestamp('2019-04-30'))


def test_seasonality_and_trend(stseries):
    S = fe.Seasonality(lags=6, tfreq='M').fit_transform(stseries)
    T = fe.Trend(lags=6, tfreq='M').fit_transform(stseries)
    _check_features(S, stseries, 'seasonal', 6, pd.Timestamp('2019-07-31'))
    _check_features(T, stseries, 'trend', 6, pd.Timestamp('2019-07-31'))
    assert not np.allclose(S.values, T.values)


def test_tfreq_is_inferred_when_omitted(stseries):
    X = fe.AR(lags=2).fit_transform(stseries)
    assert X.index.get_level_values('t').max() == pd.Timestamp('2021-01-31')


def test_validation(stseries):
    with pytest.raises(ValueError):
        fe.AR(lags=1)
    with pytest.raises(ValueError, match='lags'):
        fe.AR(lags=30, tfreq='M').fit_transform(stseries)
    with pytest.raises(ValueError, match='2 \\* lags'):
        fe.Seasonality(lags=13, tfreq='M').fit_transform(stseries)


def test_pandas_feature_union(stseries):
    union = PandasFeatureUnion([('ar', fe.AR(lags=2, tfreq='M')),
                                ('seasonal', fe.Seasonality(lags=4, tfreq='M')),
                                ('skip', None)])
    X = union.fit_transform(stseries)
    assert list(X.columns) == ['ar_1', 'ar_2', 'seasonal_1', 'seasonal_2', 'seasonal_3', 'seasonal_4']
    # rows are aligned on the intersection of the indexes (the STL warm-up wins)
    assert X.index.get_level_values('t').min() == pd.Timestamp('2019-05-31')
    assert not X.isna().any().any()
    pd.testing.assert_frame_equal(union.transform(stseries), X)
    with pytest.raises(ValueError):
        PandasFeatureUnion([]).fit_transform(stseries)


def test_feature_scaling(stseries):
    X = fe.AR(lags=2, tfreq='M').fit_transform(stseries)
    scaled = fe.FeatureScaling(StandardScaler()).fit_transform(X)
    assert isinstance(scaled, pd.DataFrame)
    assert scaled.index.equals(X.index) and list(scaled.columns) == list(X.columns)
    assert np.allclose(scaled.mean(), 0, atol=1e-8)
