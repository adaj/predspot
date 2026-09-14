import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from predspot import crime_mapping as cm
from predspot import feature_engineering as fe
from predspot import ml_modelling as ml
from predspot.utilities import PandasFeatureUnion


def make_pipeline(grid, mapping_cls=cm.KDE, estimator=None):
    return ml.PredictionPipeline(
        mapping=mapping_cls(tfreq='M', grid=grid),
        fextraction=PandasFeatureUnion([
            ('seasonal', fe.Seasonality(lags=4, tfreq='M')),
            ('trend', fe.Trend(lags=4, tfreq='M')),
            ('diff', fe.Diff(lags=4, tfreq='M')),
        ]),
        estimator=estimator or Pipeline([
            ('f_scaling', fe.FeatureScaling(QuantileTransformer(n_quantiles=10))),
            ('f_selection', ml.FeatureSelection(RFE(RandomForestRegressor(n_estimators=5, random_state=0)))),
            ('model', ml.Model(RandomForestRegressor(n_estimators=10, random_state=0))),
        ]),
        random_state=0,
    )


@pytest.fixture(scope='module')
def fitted(dataset, study_area):
    grid = cm.create_gridpoints(study_area, resolution=2)
    return make_pipeline(grid).fit(dataset)


def test_fit_predict(fitted):
    n_places = len(fitted.grid)
    assert fitted.next_time == pd.Timestamp('2021-01-31')
    pred = fitted.predict()
    assert list(pred.columns) == ['crime_density']
    assert len(pred) == n_places
    assert pred.index.get_level_values('t').unique().tolist() == [pd.Timestamp('2021-01-31')]
    assert (pred['crime_density'] >= 0).all()
    # the forecast is appended to the series and the horizon moves forward
    assert fitted.stseries.index.get_level_values('t').max() == pd.Timestamp('2021-01-31')
    assert fitted.next_time == pd.Timestamp('2021-02-28')
    pred2 = fitted.predict()
    assert pred2.index.get_level_values('t').unique().tolist() == [pd.Timestamp('2021-02-28')]


def test_feature_importances(fitted):
    fi = fitted.feature_importances
    assert list(fi.columns) == ['importance']
    assert np.isclose(fi['importance'].sum(), 1)
    assert set(fi.index) <= set(fitted.features.columns)


def test_evaluate(dataset, study_area):
    grid = cm.create_gridpoints(study_area, resolution=2)
    pipe = make_pipeline(grid).fit(dataset)
    scores = pipe.evaluate('r2', cv=3)
    assert len(scores) == 3
    mse = pipe.evaluate('mse', cv=3)
    assert all(s >= 0 for s in mse)
    with pytest.raises(ValueError):
        pipe.evaluate('mae')
    with pytest.raises(ValueError):
        pipe.evaluate('r2', cv=100)
    # evaluate refits on the full data: predictions still work afterwards
    assert len(pipe.predict()) == len(grid)


def test_plain_sklearn_estimator(dataset, study_area):
    grid = cm.create_gridpoints(study_area, resolution=2)
    pipe = make_pipeline(grid, estimator=LinearRegression()).fit(dataset)
    pred = pipe.predict()
    assert len(pred) == len(grid)
    with pytest.raises(AttributeError):
        pipe.feature_importances


def test_quadrat_count_pipeline(dataset, study_area):
    grid = cm.create_gridhexagonal(study_area, resolution=2)
    pipe = make_pipeline(grid, mapping_cls=cm.QuadratCount).fit(dataset)
    pred = pipe.predict()
    assert len(pred) == len(grid)


def test_not_fitted(study_area):
    grid = cm.create_gridpoints(study_area, resolution=2)
    with pytest.raises(RuntimeError):
        make_pipeline(grid).predict()
