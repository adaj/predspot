import pandas as pd
import pytest

from predspot import pipeline


def test_generate_testdata():
    crimes, area = pipeline.generate_testdata(500, "2019-01-01", "2019-12-31", seed=1)
    assert list(crimes.columns) == ["tag", "t", "lon", "lat"]
    assert len(crimes) == 500
    assert area.crs.to_epsg() == 4326
    assert crimes["t"].between("2019-01-01", "2019-12-31").all()
    again, _ = pipeline.generate_testdata(500, "2019-01-01", "2019-12-31", seed=1)
    pd.testing.assert_frame_equal(crimes, again)


def test_run_prediction_pipeline(crimes, study_area):
    pred, pipe = pipeline.run_prediction_pipeline(
        crimes, study_area, crime_tags=["burglary"], grid_resolution=2, random_state=0
    )
    assert len(pred) == len(pipe.grid)
    assert pred.index.get_level_values("t").unique().tolist() == [pd.Timestamp("2021-01-31")]
    scores = pipeline.evaluate_pipeline(pipe, "r2", cv=2)
    assert len(scores) == 2


def test_run_prediction_pipeline_validation(crimes, study_area):
    with pytest.raises(ValueError):
        pipeline.run_prediction_pipeline(crimes.drop(columns=["tag"]), study_area)
