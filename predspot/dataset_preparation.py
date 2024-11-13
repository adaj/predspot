"""
Dataset Preparation Module

This module provides functionality for preparing and managing crime datasets along with their
corresponding study areas. It handles spatial data processing and visualization of crime incidents.
"""

__author__ = 'Adelson Araujo'

import pandas as pd
import geopandas as gpd
import numpy as np
import warnings
from shapely.geometry import Point, LineString


class Dataset:
    """
    A class to handle crime datasets and their associated study areas.

    Args:
        crimes (pandas.DataFrame): DataFrame containing crime data with required columns:
            'tag', 't' (timestamp), 'lon' (longitude), and 'lat' (latitude)
        study_area (geopandas.GeoDataFrame): GeoDataFrame defining the study area boundaries
        debug (bool, optional): Enable debug printing. Defaults to False.

    Attributes:
        crimes (geopandas.GeoDataFrame): Processed crime data with geometry
        study_area (geopandas.GeoDataFrame): Study area boundaries
    """

    def __init__(self, crimes, study_area, debug=False):#, poi_data=None):
        self.debug = debug
        
        if self.debug:
            print("Initializing Dataset class", flush=True)
            
        assert isinstance(study_area, gpd.GeoDataFrame), \
            "study_area must be a geopandas GeoDataFrame."
        self._study_area = study_area
        
        assert isinstance(crimes, pd.DataFrame) \
               and all([x in crimes.columns for x in ['tag', 't', 'lon', 'lat']]),\
            "Input crime data must be a pandas Data Frame and " \
            + "have at least `tag`, `t`, `lon` and `lat` as columns."
            
        if self.debug:
            print(f"Processing {len(crimes)} crime incidents", flush=True)
            
        self._crimes = crimes
        self._crimes['geometry'] = self._crimes.apply(lambda x: Point([x['lon'], x['lat']]),
                                                      axis=1)
        self._crimes = gpd.GeoDataFrame(self._crimes, crs={'init': 'epsg:4326'})
        self._crimes['t'] = self._crimes['t'].apply(pd.to_datetime)
        
        if self.debug:
            print("Dataset initialization complete", flush=True)

    def __repr__(self):
        """
        String representation of the Dataset object.

        Returns:
            str: A formatted string showing dataset statistics
        """
        return 'predspot.Dataset<\n'\
             + f'  crimes = GeoDataFrame({self._crimes.shape[0]}),\n' \
             + f'    >> {self.crimes["tag"].value_counts().to_dict()}\n' \
             + f'  study_area = GeoDataFrame({self._study_area.shape[0]}),\n' \
             + '>'

    @property
    def crimes(self):
        """
        Get the crime incidents data.

        Returns:
            geopandas.GeoDataFrame: The processed crime incidents data
        """
        return self._crimes

    @property
    def study_area(self):
        """
        Get the study area boundaries.

        Returns:
            geopandas.GeoDataFrame: The study area boundaries
        """
        return self._study_area

    @property
    def shape(self):
        """
        Get the shapes of the dataset components.

        Returns:
            dict: Dictionary containing the shapes of crimes and study_area DataFrames
        """
        return {'crimes': self._crimes.shape,
                'study_area': self._study_area.shape}

    def plot(self, ax=None, crime_samples=1000, **kwargs):
        """
        Plot the study area and crime incidents.

        Args:
            ax (matplotlib.axes.Axes, optional): Matplotlib axes for plotting
            crime_samples (int, optional): Number of crime samples to plot. Defaults to 1000
            **kwargs: Additional keyword arguments for plotting
                study_area: kwargs for study area plot
                crimes: kwargs for crime incidents plot

        Returns:
            matplotlib.axes.Axes: The plot axes
        """
        if self.debug:
            print(f"Plotting dataset with {crime_samples} sample points", flush=True)
            
        if ax is None:
            ax = self.study_area.plot(color='white', edgecolor='black',
                                      **kwargs.pop('study_area',{}))
        else:
            self.study_area.plot(color='white', edgecolor='black', ax=ax,
                                 **kwargs.pop('study_area',{}))
        if crime_samples > len(self.crimes):
            crime_samples = len(self.crimes)
        self.crimes.sample(crime_samples).plot(ax=ax, marker='x',
                                               **kwargs.pop('crimes',{}))
        return ax

    def train_test_split(self, test_size=0.25):
        """
        Split the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
                Must be between 0 and 1. Defaults to 0.25.

        Returns:
            tuple: (train_dataset, test_dataset) - Two Dataset objects containing the splits

        Raises:
            AssertionError: If test_size is not between 0 and 1
        """
        if self.debug:
            print(f"Splitting dataset with test_size={test_size}", flush=True)
            
        assert 0 < test_size < 1, \
            'test_size must be between 0 and 1.'
        test_dataset = Dataset(self.crimes.sample(frac=test_size), self.study_area)
        train_ix = set(self.crimes.index) - set(test_dataset.crimes.index)
        train_dataset = Dataset(self.crimes.loc[train_ix], self.study_area)
        
        if self.debug:
            print(f"Split complete - Train size: {len(train_dataset.crimes)}, "
                  f"Test size: {len(test_dataset.crimes)}", flush=True)
            
        return train_dataset, test_dataset
