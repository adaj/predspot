"""
Dataset Preparation Module
==========================

Prepares crime event data together with the study area it belongs to.
:class:`Dataset` validates the input, converts the events to a GeoDataFrame
of points in WGS84 and offers simple plotting and splitting helpers.
"""

__author__ = "Adelson Araujo"

import logging

import geopandas as gpd
import pandas as pd

logger = logging.getLogger(__name__)

WGS84 = "EPSG:4326"
REQUIRED_COLUMNS = ("tag", "t", "lon", "lat")


class Dataset:
    """
    Crime events plus the study area they occurred in.

    Args:
        crimes (pandas.DataFrame): Crime events with at least the columns
            ``tag`` (crime type), ``t`` (timestamp, anything
            :func:`pandas.to_datetime` understands), ``lon`` and ``lat``
            (WGS84 degrees). The input is **not** modified.
        study_area (geopandas.GeoDataFrame): Boundary of the study area. It
            must have a CRS set.

    Attributes:
        crimes (geopandas.GeoDataFrame): Events as points in WGS84 with the
            original columns preserved and ``t`` parsed to datetimes.
        study_area (geopandas.GeoDataFrame): The study area boundary.
    """

    def __init__(self, crimes, study_area):
        if not isinstance(study_area, gpd.GeoDataFrame):
            raise TypeError("study_area must be a geopandas GeoDataFrame.")
        if study_area.crs is None:
            raise ValueError(
                'study_area must have a CRS set (e.g. study_area.set_crs("EPSG:4326")).'
            )
        if not isinstance(crimes, pd.DataFrame):
            raise TypeError("crimes must be a pandas DataFrame.")
        missing = [c for c in REQUIRED_COLUMNS if c not in crimes.columns]
        if missing:
            raise ValueError(
                "Input crime data must have at least `tag`, `t`, "
                f"`lon` and `lat` as columns; missing {missing}."
            )
        logger.debug("Preparing dataset with %d crime events", len(crimes))

        self._study_area = study_area
        if isinstance(crimes, gpd.GeoDataFrame) and crimes.crs is not None:
            events = gpd.GeoDataFrame(crimes.copy()).to_crs(WGS84)
        else:
            events = crimes.copy()
            events = gpd.GeoDataFrame(
                events.drop(columns=["geometry"], errors="ignore"),
                geometry=gpd.points_from_xy(events["lon"], events["lat"]),
                crs=WGS84,
            )
        events["t"] = pd.to_datetime(events["t"])
        self._crimes = events

    def __repr__(self):
        counts = self._crimes["tag"].value_counts().to_dict()
        return (
            "predspot.Dataset<\n"
            f"  crimes = GeoDataFrame({self._crimes.shape[0]}),\n"
            f"    >> {counts}\n"
            f"  study_area = GeoDataFrame({self._study_area.shape[0]}),\n"
            ">"
        )

    @property
    def crimes(self):
        """geopandas.GeoDataFrame: The crime events as points (WGS84)."""
        return self._crimes

    @property
    def study_area(self):
        """geopandas.GeoDataFrame: The study area boundary."""
        return self._study_area

    @property
    def shape(self):
        """dict: Shapes of ``crimes`` and ``study_area``."""
        return {"crimes": self._crimes.shape, "study_area": self._study_area.shape}

    def plot(self, ax=None, crime_samples=1000, **kwargs):
        """
        Plot the study area with a sample of the crime events.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to draw on.
            crime_samples (int): Number of events to draw (random sample).
            **kwargs: ``study_area=dict(...)`` and ``crimes=dict(...)`` are
                forwarded to the respective ``GeoDataFrame.plot`` calls.

        Returns:
            matplotlib.axes.Axes: The axes drawn on.
        """
        area_kwargs = {"color": "white", "edgecolor": "black"}
        area_kwargs.update(kwargs.pop("study_area", {}))
        study_area = self.study_area.to_crs(WGS84)
        ax = study_area.plot(ax=ax, **area_kwargs)
        n = min(crime_samples, len(self.crimes))
        crimes_kwargs = {"marker": "x"}
        crimes_kwargs.update(kwargs.pop("crimes", {}))
        self.crimes.sample(n).plot(ax=ax, **crimes_kwargs)
        return ax

    def train_test_split(self, test_size=0.25, random_state=None):
        """
        Randomly split the events into two datasets sharing the study area.

        Args:
            test_size (float): Fraction of events in the test dataset (0-1).
            random_state (int, optional): Seed for reproducibility.

        Returns:
            tuple: ``(train_dataset, test_dataset)``.
        """
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        test = self.crimes.sample(frac=test_size, random_state=random_state)
        train = self.crimes.drop(index=test.index)
        logger.debug("Split dataset: train=%d, test=%d", len(train), len(test))
        return Dataset(train, self.study_area), Dataset(test, self.study_area)
