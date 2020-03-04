import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline

import predspot.dataset_preparation
import predspot.feature_engineering

crime_tags = {'todas_ocorrencias': ['ASSALTO', 'OUTROS', 'AMEACAS', 'CONSUMO_DE_DROGAS',
                                    'VEICULOS_ABERTOS', 'ACIDENTES', 'FURTO_DE_TERCEIROS']}
g_resolution = 0.1

city = gpd.read_file('ufrn/ufrn.geojson')
cdata = pd.read_csv('ufrn/crime_data.csv', index_col=0)

todas_ocs = cdata.loc[cdata['tag'].isin(crime_tags['todas_ocorrencias'])]
dataset = predspot.dataset_preparation.Dataset(crimes=todas_ocs,
                                               study_area=city,
                                               grid_resolution=g_resolution)
#dataset.plot()

train, test = dataset.train_test_split(test_size=0.25)

# stseries = predspot.feature_engineering.KDE(tfreq='M', grid=dataset.grid).transform(data_points=dataset.crimes)
# stseries.head()

pipeline = Pipeline([
    ('mapping', predspot.feature_engineering.KDE(tfreq='M', grid=dataset.grid)),
    ('feature_extraction', predspot.feature_engineering.FeatureUnion([
        ('seasonal_trend', predspot.feature_engineering.STLFeatures(lags=2, offset=1)),
        ('diff', predspot.feature_engineering.DiffFeatures(lags=2)) # geographic features, and so on...
    ]))
])
X = pipeline.fit_transform(dataset.crimes)
X.head()
#
#
#

class FeatureScaling(TransformerMixin, BaseEstimator):
    pass


class FeatureSelection(TransformerMixin, BaseEstimator):
    pass


class Model(RegressorMixin, BaseEstimator):
    pass
