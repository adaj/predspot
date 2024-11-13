Predspot
========

Overview
--------

Predspot is an early project for a Python library for spatio-temporal crime prediction and hotspot detection. It combines machine learning techniques with spatial analysis to help predict and visualize crime patterns across time and space. To properly use this library, you preferably need to have a crime dataset and a study area.

The results of this library are not guaranteed to be good, but it is a good starting point for spatio-temporal crime prediction. This project is barely maintained, so please if you want to collaborate, open an issue or a PR.

Key Features
------------
* Spatial and temporal crime mapping
* Feature engineering for time series data
* Machine learning-based prediction pipeline
* Crime hotspot detection using Kernel Density Estimation
* Visualization tools for crime patterns

Quick Start
-----------

Basic usage example:

.. code-block:: python

    from predspot import Dataset, PredictionPipeline
    from predspot.crime_mapping import KDE, create_gridpoints
    from predspot.feature_engineering import Seasonality, Trend, Diff

    # Load and prepare data
    dataset = Dataset(crimes_df, study_area_gdf)

    # Create prediction pipeline
    pipeline = PredictionPipeline(
        mapping=KDE(tfreq='M', grid=create_gridpoints(study_area, resolution=250)),
        fextraction=PandasFeatureUnion([
            ('seasonal', Seasonality(lags=12)),
            ('trend', Trend(lags=12)),
            ('diff', Diff(lags=12))
        ]),
        estimator=your_favorite_sklearn_model
    )

    # Fit and predict
    pipeline.fit(dataset)
    predictions = pipeline.predict()

Modules
-------

The library consists of four main modules:

* :doc:`modules/dataset_preparation`: Module for preparing and managing crime datasets and study areas.
* :doc:`modules/crime_mapping`: Module for spatial and temporal crime mapping, including KDE-based hotspot detection.
* :doc:`modules/feature_engineering`: Module for time series feature engineering, including seasonality, trend, and difference features.
* :doc:`modules/ml_modelling`: Module that implements the prediction pipeline and model evaluation.

And two utilities:

* :doc:`modules/utilities`: Utility functions for data preparation and visualization.
* :doc:`modules/pipeline`: Functions for the prediction pipeline.

Installation
------------

Create conda env and install requirements:

.. code-block:: bash

    conda create -n predspot python=3.8
    conda activate predspot
    conda install -y rtree geopandas  # if doesnt work, do: `conda clean --all`
    pip install pandas statsmodels==0.10.2 geojsoncontour stldecompose scikit-learn matplotlib descartes
    pip install .

Required dependencies:

* pandas
* geopandas
* numpy
* scikit-learn
* scipy
* stldecompose
* matplotlib

Input Data Format
-----------------

The crime data should be a pandas DataFrame with the following required columns:

* ``tag``: Crime type
* ``t``: Timestamp
* ``lon``: Longitude
* ``lat``: Latitude

The study area should be a GeoDataFrame defining the geographical boundaries of interest.

Resources
---------

For more information on the methods used in Predspot, please search more about these methods:

* Kernel Density Estimation for crime hotspot detection
* Time series decomposition for feature engineering
* Spatio-temporal crime prediction techniques

License
-------

BSD 3-Clause.

Contributing
------------

Contributions are welcome! Please feel free to submit a Pull Request.

Guidelines for contributing:

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

Citation
--------

If you use Predspot in your research, please cite us:

.. code-block:: text

    Araujo, A., & Cacho, N. (2019). Predspot: Predicting crime hotspots with machine learning. 
    Master's thesis, UFRN (Universidade Federal do Rio Grande do Norte), Natal, Brazil.

    Araújo, A., Cacho, N., Bezerra, L., Vieira, C., & Borges, J. (2018, June). 
    Towards a crime hotspot detection framework for patrol planning. 
    In 2018 IEEE 20th International Conference on High Performance Computing and Communications; 
    IEEE 16th International Conference on Smart City; 
    IEEE 4th International Conference on Data Science and Systems (HPCC/SmartCity/DSS) (pp. 1256-1263). IEEE.

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   modules/crime_mapping
   modules/dataset_preparation
   modules/feature_engineering
   modules/ml_modelling
   modules/utilities
   modules/pipeline

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`