"""
Machine Learning Modelling Module
=================================

The prediction pipeline of Predspot and thin wrappers that make scikit-learn
feature selectors and regressors keep pandas indexes, so predictions stay
attached to their ``(t, places)`` labels.
"""

__author__ = "Adelson Araujo"

import logging

import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from predspot.crime_mapping import tfreq_offset

logger = logging.getLogger(__name__)

idx = pd.IndexSlice

SCORERS = {"r2": r2_score, "mse": mean_squared_error}


class FeatureSelection(TransformerMixin, BaseEstimator):
    """
    Wrap a scikit-learn feature selector so that it returns DataFrames.

    Args:
        estimator: A selector exposing ``support_`` after fit (e.g. ``RFE``).
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y=None):
        self.estimator.fit(x, y)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return getattr(self, "is_fitted_", False)

    @property
    def support_(self):
        """numpy.ndarray: Boolean mask of the selected columns."""
        return self.estimator.support_

    def transform(self, x):
        return pd.DataFrame(
            self.estimator.transform(x), index=x.index, columns=x.columns[self.estimator.support_]
        )


class Model(RegressorMixin, BaseEstimator):
    """
    Wrap a scikit-learn regressor so that predictions come back as DataFrames.

    Args:
        estimator: Any scikit-learn regressor.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y=None):
        self.estimator.fit(x, y)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return getattr(self, "is_fitted_", False)

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    def predict(self, x):
        return pd.DataFrame(self.estimator.predict(x), index=x.index, columns=["crime_density"])


class PredictionPipeline(RegressorMixin, BaseEstimator):
    """
    End-to-end crime hotspot prediction.

    The pipeline chains three stages: a spatio-temporal ``mapping`` (e.g.
    :class:`predspot.crime_mapping.KDE`) that turns events into a series per
    place and period; a feature extraction step (e.g.
    :class:`predspot.utilities.PandasFeatureUnion` of lag features) and a
    scikit-learn ``estimator`` (or ``Pipeline``) that learns to predict the
    next period's value from the features.

    Args:
        mapping: A :class:`predspot.crime_mapping.SpatioTemporalMapping`.
        fextraction: A transformer taking the series and returning features.
        estimator: A scikit-learn regressor or ``Pipeline`` whose last step
            returns a DataFrame with a ``crime_density`` column (see
            :class:`Model`).
        random_state (int, optional): Seed used to shuffle the training rows.
    """

    def __init__(self, mapping, fextraction, estimator, random_state=None):
        self.mapping = mapping
        self.fextraction = fextraction
        self.estimator = estimator
        self.random_state = random_state
        self._offset = tfreq_offset(mapping.tfreq)
        self._stseries = None
        self._dataset = None
        self._X = None
        self._t_plus_one = None

    @property
    def grid(self):
        """GeoDataFrame: Spatial grid used by the mapping."""
        return self.mapping._grid

    @property
    def dataset(self):
        """Dataset: The dataset the pipeline was fitted on."""
        return self._dataset

    @property
    def stseries(self):
        """pandas.Series: Spatio-temporal series (observed + predicted periods)."""
        return self._stseries

    @property
    def features(self):
        """pandas.DataFrame: Features of every ``(t, places)`` row."""
        return self._X

    @property
    def next_time(self):
        """pandas.Timestamp: The period that the next ``predict`` call forecasts."""
        return self._t_plus_one

    def _check_fitted(self):
        if self._X is None:
            raise RuntimeError("This pipeline was not fitted yet.")

    @property
    def feature_importances(self):
        """
        Importance of each selected feature.

        Works when ``estimator`` is a ``Pipeline`` whose last step exposes
        ``feature_importances_`` (e.g. :class:`Model` around a random
        forest); an optional :class:`FeatureSelection` step before it is
        taken into account.

        Returns:
            pandas.DataFrame: Importance per feature, sorted descending.
        """
        self._check_fitted()
        steps = getattr(self.estimator, "steps", [("model", self.estimator)])
        model = steps[-1][1]
        try:
            importances = model.feature_importances_
        except AttributeError as exc:
            raise AttributeError("The estimator does not expose feature_importances_.") from exc
        columns = self._X.columns
        for _, step in steps[:-1]:
            if hasattr(step, "support_"):
                columns = columns[step.support_]
        return pd.DataFrame({"importance": importances}, index=columns).sort_values(
            "importance", ascending=False
        )

    def fit(self, dataset, y=None):
        """
        Fit the mapping, the features and the estimator on a dataset.

        Args:
            dataset (predspot.Dataset): Crime events and study area.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            PredictionPipeline: ``self``.
        """
        logger.debug("Fitting prediction pipeline")
        self._dataset = dataset
        self._stseries = self.mapping.fit_transform(dataset.crimes)
        self._X = self.fextraction.fit_transform(self._stseries)
        t0 = self._X.index.get_level_values("t").min()
        tf = self._stseries.index.get_level_values("t").max()
        X = self._X.loc[t0:tf].sample(frac=1, random_state=self.random_state)
        y = self._stseries.loc[X.index]
        self.estimator.fit(X, y)
        self._t_plus_one = self._X.index.get_level_values("t").max()
        logger.debug("Pipeline fitted on %d rows; next period is %s", len(X), self._t_plus_one)
        return self

    def predict(self):
        """
        Forecast the next period for every place.

        Each call appends its forecast to the series and recomputes the
        features, so calling it repeatedly walks forward in time.

        Returns:
            pandas.DataFrame: ``crime_density`` indexed by ``(t, places)``
            for the forecast period.
        """
        self._check_fitted()
        X = self._X.loc[[self._t_plus_one], :]
        y_pred = pd.DataFrame(self.estimator.predict(X), index=X.index)
        y_pred.columns = ["crime_density"]
        logger.debug("Predicted %d places for %s", len(y_pred), self._t_plus_one)
        self._stseries = pd.concat([self._stseries, y_pred["crime_density"]]).sort_index()
        self._stseries.name = "crime_density"
        self._X = self.fextraction.transform(self._stseries)
        self._t_plus_one = self._t_plus_one + self._offset
        return y_pred

    def evaluate(self, scoring="r2", cv=5):
        """
        Score the estimator with time series cross-validation.

        Periods are split in ``cv`` consecutive folds
        (:class:`sklearn.model_selection.TimeSeriesSplit`); the estimator is
        refitted on the original data afterwards.

        Args:
            scoring (str): ``'r2'`` or ``'mse'``.
            cv (int): Number of folds (must be lower than the number of periods).

        Returns:
            list: One score per fold.
        """
        self._check_fitted()
        if scoring not in SCORERS:
            raise ValueError('invalid scoring. Try "r2" or "mse".')
        scorer = SCORERS[scoring]
        timestamps = (
            self._X.index.get_level_values("t")
            .unique()
            .intersection(self._stseries.index.get_level_values("t").unique())
            .sort_values()
        )
        if not isinstance(cv, int) or cv >= len(timestamps):
            raise ValueError("cv must be an integer lower than the number of periods.")
        scores = []
        for train_t, test_t in TimeSeriesSplit(cv).split(timestamps):
            X_train = self._X.loc[idx[timestamps[train_t], :], :].sample(
                frac=1, random_state=self.random_state
            )
            X_test = self._X.loc[idx[timestamps[test_t], :], :]
            y_train = self._stseries.loc[X_train.index]
            y_test = self._stseries.loc[X_test.index]
            self.estimator.fit(X_train, y_train)
            y_pred = self.estimator.predict(X_test)
            scores.append(scorer(y_test, y_pred))
        logger.debug("%s-fold CV %s scores: %s", cv, scoring, scores)
        self.fit(self._dataset)  # back to normal
        return scores
