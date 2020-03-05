"""
Author: Adelson Araujo
"""

from pandas import DatetimeIndex
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

from predspot.dataset_preparation import Dataset
from predspot.feature_engineering import KDE, PandasFeatureUnion, STLFeatures, DiffFeatures
from predspot.ml_modelling import FeatureScaling, FeatureSelection, Model, PredictionPipeline


def load(study_area, crime_data, crime_tags, between_time, tfreq, grid_resolution):
    """1. Dataset Preparation"""
    # Filtering crime_tags
    cdata = crime_data.loc[crime_data['tag'].isin(crime_tags)]
    # Filtering between_time
    time_ix = DatetimeIndex(cdata['t']).indexer_between_time(between_time[0],between_time[1])
    cdata = cdata.iloc[time_ix]

    dataset = Dataset(cdata, study_area, grid_resolution)

    pred_pipeline = PredictionPipeline(
        mapping=KDE(tfreq=tfreq, grid=dataset.grid),
        fextraction=PandasFeatureUnion([
            ('seasonal_trend', STLFeatures(lags=2, offset=1)),
            ('diff', DiffFeatures(lags=2))
            # geographic features
        ]),
        estimator=Pipeline([
            ('f_scaling', FeatureScaling(QuantileTransformer(10, output_distribution='uniform'))),
            ('f_selection', FeatureSelection(RFE(RandomForestRegressor()))),
            ('model', Model(RandomForestRegressor(50)))
        ])
    )
    pred_pipeline.fit(dataset)
    return pred_pipeline
