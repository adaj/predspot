"""
Feature Engineering Module
==========================

Turns the spatio-temporal series produced by a mapping (see
:mod:`predspot.crime_mapping`) into lagged features, one row per
``(t, places)``. Each feature class applies a time series transformation to
the history of every place and then builds ``lags`` lagged columns from it:

* :class:`AR` — the raw series (autoregressive features);
* :class:`Diff` — first differences;
* :class:`Seasonality` — the seasonal component of an STL decomposition;
* :class:`Trend` — the trend component of an STL decomposition.

The output always contains one extra row for the period right after the last
observed one, so that the fitted model can forecast the next period.
"""

__author__ = "Adelson Araujo"

import logging
from abc import abstractmethod

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import STL

from predspot.crime_mapping import tfreq_offset

logger = logging.getLogger(__name__)


def infer_offset(time_index):
    """
    Infer the :class:`pandas.DateOffset` between consecutive periods.

    Args:
        time_index (pandas.DatetimeIndex): Unique, sorted period labels.

    Returns:
        pandas.DateOffset: The offset separating consecutive periods.

    Raises:
        ValueError: If the frequency cannot be inferred (e.g. fewer than 3
            periods); pass ``tfreq`` explicitly in that case.
    """
    time_index = pd.DatetimeIndex(time_index).unique().sort_values()
    freq = pd.infer_freq(time_index) if len(time_index) >= 3 else None
    if freq is None:
        raise ValueError(
            "Could not infer the time frequency of the series; pass `tfreq` explicitly."
        )
    return pd.tseries.frequencies.to_offset(freq)


class TimeSeriesFeatures(BaseEstimator, TransformerMixin):
    """
    Base class for lagged time series features.

    Args:
        lags (int): Number of lagged columns to create (> 1).
        tfreq (str, optional): Time frequency of the series (``'M'``, ``'W'``
            or ``'D'``). If omitted it is inferred from the series index.
    """

    def __init__(self, lags, tfreq=None):
        if not isinstance(lags, int) or lags < 2:
            raise ValueError("`lags` must be an integer greater than 1.")
        self.lags = lags
        self.tfreq = tfreq
        self._offset = tfreq_offset(tfreq) if tfreq is not None else None

    @property
    def label(self):
        """str: Prefix of the feature columns (override in subclasses)."""
        return "feature"

    @abstractmethod
    def apply_ts_decomposition(self, ts):
        """
        Transform the time series of one place before lagging it.

        Args:
            ts (pandas.Series): Series of one place indexed by time.

        Returns:
            pandas.Series: Transformed series.
        """

    def fit(self, x=None, y=None):
        """No-op; present for scikit-learn compatibility."""
        return self

    def make_lag_df(self, ts):
        """
        Build the lagged feature columns of one time series.

        Args:
            ts (pandas.Series): Series indexed by time.

        Returns:
            tuple: ``(lag_df, ts_aligned)`` — the lagged features and the
            original series restricted to the same index.
        """
        if len(ts) <= self.lags:
            raise ValueError("`lags` is higher than the number of time periods.")
        lag_df = pd.concat([ts.shift(lag) for lag in range(1, self.lags + 1)], axis=1)
        lag_df = lag_df.iloc[self.lags :]
        lag_df.columns = [f"{self.label}_{i}" for i in range(1, self.lags + 1)]
        return lag_df, ts.loc[lag_df.index]

    def transform(self, stseries):
        """
        Compute lagged features for every place.

        Args:
            stseries (pandas.Series): Series indexed by ``(t, places)``.

        Returns:
            pandas.DataFrame: Features indexed by ``(t, places)``, including
            one row for the period after the last observed one.
        """
        times = stseries.index.get_level_values("t")
        offset = self._offset if self._offset is not None else infer_offset(times)
        places = stseries.index.get_level_values("places").unique()
        logger.debug(
            "%s: computing %d lags for %d places", type(self).__name__, self.lags, len(places)
        )
        frames = []
        for place in places:
            ts = stseries.xs(place, level="places").sort_index()
            ts = self.apply_ts_decomposition(ts)
            ts.loc[ts.index[-1] + offset] = None  # next period, to be forecast
            f, _ = self.make_lag_df(ts)
            f["places"] = place
            frames.append(f.set_index("places", append=True))
        X = pd.concat(frames)
        X.index.names = ["t", "places"]
        return X.sort_index()


class AR(TimeSeriesFeatures):
    """Autoregressive features: lags of the raw series."""

    @property
    def label(self):
        return "ar"

    def apply_ts_decomposition(self, ts):
        return ts


class Diff(TimeSeriesFeatures):
    """Lags of the first difference of the series."""

    @property
    def label(self):
        return "diff"

    def apply_ts_decomposition(self, ts):
        return ts.diff().iloc[1:]


class _STLFeatures(TimeSeriesFeatures):
    """Shared STL decomposition; ``lags`` is also used as the STL period."""

    component = None

    def apply_ts_decomposition(self, ts):
        if len(ts) < 2 * self.lags:
            raise ValueError(
                f"{type(self).__name__} needs at least 2 * lags "
                f"({2 * self.lags}) periods; got {len(ts)}."
            )
        result = STL(ts, period=self.lags).fit()
        return getattr(result, self.component)


class Seasonality(_STLFeatures):
    """Lags of the seasonal component of an STL decomposition (period = lags)."""

    component = "seasonal"

    @property
    def label(self):
        return "seasonal"


class Trend(_STLFeatures):
    """Lags of the trend component of an STL decomposition (period = lags)."""

    component = "trend"

    @property
    def label(self):
        return "trend"


class FeatureScaling(TransformerMixin, BaseEstimator):
    """
    Wrap a scikit-learn scaler so that it returns DataFrames.

    Args:
        estimator: Any scikit-learn transformer (e.g. ``QuantileTransformer``).
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y=None):
        self.estimator.fit(x, y)
        self.is_fitted_ = True
        return self

    def __sklearn_is_fitted__(self):
        return getattr(self, "is_fitted_", False)

    def transform(self, x):
        return pd.DataFrame(self.estimator.transform(x), index=x.index, columns=x.columns)
