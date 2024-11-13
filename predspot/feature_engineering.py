"""
Feature Engineering Module

This module provides classes for time series feature engineering and transformation,
including autoregressive features, differencing, seasonality, and trend decomposition.
"""

__author__ = 'Adelson Araujo'

from abc import abstractmethod
from numpy import vstack
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from stldecompose import decompose


class TimeSeriesFeatures(BaseEstimator, TransformerMixin):
    """
    Base class for time series feature engineering.

    Args:
        lags (int): Number of time lags to use for feature creation
        tfreq (str): Time frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        debug (bool, optional): Enable debug printing. Defaults to False

    Raises:
        AssertionError: If lags is not a positive integer or tfreq is invalid
    """

    def __init__(self, lags, tfreq, debug=False):
        self.debug = debug
        if self.debug:
            print(f"Initializing TimeSeriesFeatures with lags={lags}, freq={tfreq}", flush=True)
            
        assert isinstance(lags, int) and lags > 1, \
            '`lags` must be a positive integer.'
        self._lags = lags
        assert tfreq in ['D', 'W', 'M'], \
            '`tfreq` not allowed, choose between `D`, `W`, `M`.'
        self._tfreq = tfreq

    @property
    def lags(self):
        """int: Number of time lags"""
        return self._lags

    @property
    def label(self):
        """str: Feature label identifier"""
        return 'feature'  # override this label if implementing a new feature

    @abstractmethod
    def apply_ts_decomposition(self, ts):
        """
        Apply time series decomposition.

        Args:
            ts (pandas.Series): Input time series

        Returns:
            pandas.Series: Transformed time series
        """
        pass

    def make_lag_df(self, ts):
        """
        Create lagged features dataframe.

        Args:
            ts (pandas.Series): Input time series

        Returns:
            tuple: (lag_df, aligned_ts) - Lagged features and aligned original series

        Raises:
            AssertionError: If series length is less than number of lags
        """
        if self.debug:
            print(f"Creating lag features for series of length {len(ts)}", flush=True)
            
        assert len(ts) > self.lags, "`lags` are higher than temporal units."
        lag_df = pd.concat([ts.shift(lag) for lag in range(1, self.lags+1)], axis=1)
        lag_df = lag_df.iloc[self.lags:]
        lag_df.columns = ['{}_{}'.format(self.label, i) for i in range(1, self.lags+1)]
        return lag_df, ts.loc[lag_df.index]

    def transform(self, stseries):
        """
        Transform the input series into lagged features.

        Args:
            stseries (pandas.Series): Input time series with multi-index (time, places)

        Returns:
            pandas.DataFrame: Transformed features
        """
        if self.debug:
            print(f"Transforming series with {len(stseries)} observations", flush=True)
            
        X = pd.DataFrame()
        if self._tfreq=='M':
            next_time = pd.tseries.offsets.MonthEnd(1)
        elif self._tfreq=='W':
            next_time = pd.tseries.offsets.Week(1)
        elif self._tfreq=='D':
            next_time = pd.tseries.offsets.Day(1)
            
        places = stseries.index.get_level_values('places').unique()
        for place in places:
            if self.debug:
                print(f"Processing features for place: {place}", flush=True)
                
            ts = stseries.loc[pd.IndexSlice[:, place]]
            ts = self.apply_ts_decomposition(ts)
            ts.loc[ts.index[-1] + next_time] = None
            f, _ = self.make_lag_df(ts)
            f['places'] = place
            f = f.set_index('places', append=True)
            X = X.append(f)
        X = X.sort_index()
        return X


class AR(TimeSeriesFeatures):
    """
    Autoregressive features implementation.
    """

    @property
    def label(self):
        """str: Feature label for autoregressive features"""
        return 'ar'

    def apply_ts_decomposition(self, ts):
        """
        Apply autoregressive transformation (identity).

        Args:
            ts (pandas.Series): Input time series

        Returns:
            pandas.Series: Original time series
        """
        return ts


class Diff(TimeSeriesFeatures):
    """
    Difference features implementation.
    """

    @property
    def label(self):
        """str: Feature label for difference features"""
        return 'diff'

    def apply_ts_decomposition(self, ts):
        """
        Apply difference transformation.

        Args:
            ts (pandas.Series): Input time series

        Returns:
            pandas.Series: Differenced time series
        """
        return ts.diff()[1:]


class Seasonality(TimeSeriesFeatures):
    """
    Seasonal decomposition features implementation.
    """

    @property
    def label(self):
        """str: Feature label for seasonal features"""
        return 'seasonal'

    def apply_ts_decomposition(self, ts):
        """
        Extract seasonal component from time series.

        Args:
            ts (pandas.Series): Input time series

        Returns:
            pandas.Series: Seasonal component
        """
        if self.debug:
            print(f"Extracting seasonality with period={self._lags}", flush=True)
        return decompose(ts, period=self._lags).seasonal


class Trend(TimeSeriesFeatures):
    """
    Trend decomposition features implementation.
    """

    @property
    def label(self):
        """str: Feature label for trend features"""
        return 'trend'

    def apply_ts_decomposition(self, ts):
        """
        Extract trend component from time series.

        Args:
            ts (pandas.Series): Input time series

        Returns:
            pandas.Series: Trend component
        """
        if self.debug:
            print(f"Extracting trend with period={self._lags}", flush=True)
        return decompose(ts, period=self._lags).trend


class FeatureScaling(TransformerMixin, BaseEstimator):
    """
    Feature scaling transformer.

    Args:
        estimator: Scikit-learn compatible scaling estimator
        debug (bool, optional): Enable debug printing. Defaults to False
    """

    def __init__(self, estimator, debug=False):
        self.debug = debug
        self._estimator = estimator
        
        if self.debug:
            print("Initializing FeatureScaling", flush=True)

    def transform(self, x):
        """
        Transform features using the scaling estimator.

        Args:
            x (pandas.DataFrame): Input features

        Returns:
            pandas.DataFrame: Scaled features
        """
        if self.debug:
            print(f"Scaling features of shape {x.shape}", flush=True)
            
        return pd.DataFrame(
            self._estimator.transform(x), 
            index=x.index,
            columns=x.columns
        )
