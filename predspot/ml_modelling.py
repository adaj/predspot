"""
Machine Learning Modelling Module

This module provides classes for machine learning model pipelines, feature selection,
and prediction functionality for crime density forecasting.
"""

__author__ = 'Adelson Araujo'

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

idx = pd.IndexSlice


class FeatureSelection(TransformerMixin, BaseEstimator):
    """
    Feature selection transformer.

    Args:
        estimator: Scikit-learn compatible feature selector
        debug (bool, optional): Enable debug printing. Defaults to False
    """

    def __init__(self, estimator, debug=False):
        self._estimator = estimator
        self.debug = debug
        
        if self.debug:
            print("Initializing FeatureSelection", flush=True)

    def fit(self, x, y=None):
        """
        Fit the feature selector.

        Args:
            x (pandas.DataFrame): Input features
            y (pandas.Series, optional): Target variable

        Returns:
            self: The fitted instance
        """
        if self.debug:
            print(f"Fitting feature selector with {x.shape[1]} features", flush=True)
            
        self._estimator.fit(x, y)
        return self

    def transform(self, x):
        """
        Transform features using the feature selector.

        Args:
            x (pandas.DataFrame): Input features

        Returns:
            pandas.DataFrame: Selected features
        """
        if self.debug:
            print(f"Transforming features, selecting {sum(self._estimator.support_)} features", flush=True)
            
        return pd.DataFrame(
            self._estimator.transform(x), 
            index=x.index,
            columns=x.columns[self._estimator.support_]
        )


class Model(RegressorMixin, BaseEstimator):
    """
    Model wrapper for crime density prediction.

    Args:
        estimator: Scikit-learn compatible regression estimator
        debug (bool, optional): Enable debug printing. Defaults to False
    """

    def __init__(self, estimator, debug=False):
        self._estimator = estimator
        self.debug = debug
        
        if self.debug:
            print("Initializing Model wrapper", flush=True)

    def fit(self, x, y=None):
        """
        Fit the regression model.

        Args:
            x (pandas.DataFrame): Input features
            y (pandas.Series, optional): Target variable

        Returns:
            self: The fitted instance
        """
        if self.debug:
            print(f"Fitting model with {x.shape[1]} features", flush=True)
            
        self._estimator.fit(x, y)
        return self

    def predict(self, x):
        """
        Make predictions using the fitted model.

        Args:
            x (pandas.DataFrame): Input features

        Returns:
            pandas.DataFrame: Predictions with 'crime_density' column
        """
        if self.debug:
            print(f"Making predictions for {len(x)} instances", flush=True)
            
        return pd.DataFrame(
            self._estimator.predict(x), 
            index=x.index,
            columns=['crime_density']
        )


class PredictionPipeline(RegressorMixin, BaseEstimator):
    """
    Complete pipeline for crime density prediction.

    Args:
        mapping: Spatial mapping transformer
        fextraction: Feature extraction transformer
        estimator: Scikit-learn compatible pipeline or estimator
        debug (bool, optional): Enable debug printing. Defaults to False
    """

    def __init__(self, mapping, fextraction, estimator, debug=False):
        self._mapping = mapping
        self._fextraction = fextraction
        self._estimator = estimator
        self._stseries = None
        self._dataset = None
        self.debug = debug
        
        if self.debug:
            print("Initializing PredictionPipeline", flush=True)
            
        if mapping._tfreq == 'M':
            self._offset = pd.tseries.offsets.MonthEnd(1)
        elif mapping._tfreq == 'W':
            self._offset = pd.tseries.offsets.Week(1)
        elif mapping._tfreq == 'D':
            self._offset = pd.tseries.offsets.Day(1)

    @property
    def grid(self):
        """GeoDataFrame: Spatial grid used for mapping"""
        return self._mapping._grid

    @property
    def dataset(self):
        """Dataset: Current dataset being used"""
        return self._dataset

    @property
    def stseries(self):
        """pandas.Series: Spatio-temporal series"""
        return self._stseries

    @property
    def feature_importances(self):
        """
        Get feature importance scores.

        Returns:
            pandas.DataFrame: Feature importance scores

        Raises:
            Exception: If model hasn't been fitted or doesn't support feature importances
        """
        assert self._X is not None, 'this instance was not fitted yet.'
        try:
            return pd.DataFrame(
                self._estimator.steps[-1][1]._estimator.feature_importances_,
                index=self._X.columns[self._estimator.steps[-2][1]._estimator.support_],
                columns=['importance']
            )
        except:
            raise Exception('estimator used has not feature importances implemented yet.')

    def evaluate(self, scoring, cv=5):
        """
        Evaluate model performance using time series cross-validation.

        Args:
            scoring (str): Scoring metric ('r2' or 'mse')
            cv (int): Number of cross-validation folds

        Returns:
            list: Scores for each fold

        Raises:
            Exception: If scoring metric is invalid
        """
        if self.debug:
            print(f"Evaluating model with {cv}-fold time series CV", flush=True)
            
        assert self._X is not None, 'this instance was not fitted yet.'
        if scoring == 'r2':
            scoring = r2_score
        elif scoring == 'mse':
            scoring = mean_squared_error
        else:
            raise Exception('invalid scoring. Try "r2" or "mse".')
            
        timestamps = self._X.index.get_level_values('t').unique()\
            .intersection(self._stseries.index.get_level_values('t').unique())
        assert isinstance(cv, int) and cv < len(timestamps), \
            'cv must be an integer and not higher than the number of timestamps available.'
            
        scores = []
        for train_t, test_t in TimeSeriesSplit(cv).split(timestamps):
            if self.debug:
                print(f"CV fold - train size: {len(train_t)}, test size: {len(test_t)}", flush=True)
                
            X_train = self._X.loc[idx[timestamps[train_t], :], :].sample(frac=1)
            X_test = self._X.loc[idx[timestamps[test_t], :], :]
            y_train = self._stseries.loc[X_train.index]
            y_test = self._stseries.loc[timestamps[test_t]]
            self._estimator.fit(X_train, y_train)
            y_pred = self._estimator.predict(X_test)
            scores.append(scoring(y_test, y_pred))
            
        self.fit(self._dataset)  # back to normal
        return scores

    def fit(self, dataset, y=None):
        """
        Fit the complete prediction pipeline.

        Args:
            dataset: Input dataset containing crimes and study area
            y: Ignored, present for scikit-learn compatibility

        Returns:
            self: The fitted instance

        Raises:
            Exception: If fitting fails
        """
        if self.debug:
            print("Fitting prediction pipeline", flush=True)
            
        self._dataset = dataset
        self._stseries = self._mapping.fit_transform(dataset.crimes)
        self._X = self._fextraction.fit_transform(self._stseries)
        t0 = self._X.index.get_level_values('t').unique().min()
        tf = self._stseries.index.get_level_values('t').unique().max()
        X = self._X.loc[t0:tf].sample(frac=1)  # shuffle for training
        y = self._stseries.loc[X.index]        # shuffle for training
        
        try:
            self._estimator.fit(X, y)
        except Exception as e:
            raise Exception(f'ERROR: {t0}, {tf} \nX: {self._X.loc[t0:tf]}') from e
            
        self._t_plus_one = self._X.index.get_level_values('t').unique()[-1]
        return self

    def predict(self):
        """
        Make predictions for the next time step.

        Returns:
            pandas.DataFrame: Predictions for next time step
        """
        if self.debug:
            print(f"Predicting for time step: {self._t_plus_one}", flush=True)
            
        X = self._X.loc[[self._t_plus_one],:]
        y_pred = self._estimator.predict(X)
        y_pred = pd.DataFrame(y_pred, index=X.index)
        self._stseries = self._stseries.append(y_pred['crime_density'])
        self._X = self._fextraction.transform(self._stseries)
        self._t_plus_one += self._offset
        return y_pred
