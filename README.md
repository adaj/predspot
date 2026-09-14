# Predspot

## Overview 📖

Predspot is a Python library for spatio-temporal crime prediction and hotspot detection. It combines machine learning techniques with spatial analysis to help predict and visualize crime patterns across time and space.

Key features:
- Spatial and temporal crime mapping
- Feature engineering for time series data
- Machine learning-based prediction pipeline
- Crime hotspot detection using Kernel Density Estimation
- Visualization tools for crime patterns

## Status 🚧

Predspot started as part of a master's thesis (2018-2019) and is being revived
and modernised. The code base now targets Python 3.10+ with current versions of
pandas (>= 2.2), GeoPandas (>= 1.0), scikit-learn and statsmodels, and is
covered by a test suite. It remains research software: use it as a reference
implementation and adapt it to your own data.

## How to use? 🚀

> Documentation under construction.

Basic usage example:

```python
from predspot import Dataset, PredictionPipeline
from predspot.crime_mapping import KDE, create_gridpoints
from predspot.feature_engineering import Seasonality, Trend, Diff

from predspot.utilities import PandasFeatureUnion
from sklearn.ensemble import RandomForestRegressor

# Load and prepare data: crimes_df needs `tag`, `t`, `lon`, `lat` columns and
# study_area_gdf is a GeoDataFrame with the boundary of the study area
dataset = Dataset(crimes_df, study_area_gdf)

# Create prediction pipeline (monthly KDE on a 1 km point grid)
pipeline = PredictionPipeline(
    mapping=KDE(tfreq='M', grid=create_gridpoints(study_area_gdf, resolution=1)),
    fextraction=PandasFeatureUnion([
        ('seasonal', Seasonality(lags=12)),
        ('trend', Trend(lags=12)),
        ('diff', Diff(lags=12))
    ]),
    estimator=RandomForestRegressor()
)

# Fit and predict the next month for every grid point
pipeline.fit(dataset)
predictions = pipeline.predict()
```

Prefer counting events per cell instead of a density surface? Use the
hexagonal grid with `QuadratCount`:

```python
from predspot.crime_mapping import QuadratCount, create_gridhexagonal

mapping = QuadratCount(tfreq='W', grid=create_gridhexagonal(study_area_gdf, resolution=1))
```

Or run the default pipeline in one call:

```python
from predspot.pipeline import generate_testdata, run_prediction_pipeline

crimes, study_area = generate_testdata(2000, '2019-01-01', '2020-12-31', seed=0)
predictions, pipeline = run_prediction_pipeline(crimes, study_area, grid_resolution=1)
print(pipeline.evaluate('r2', cv=3))
```


## Development ⚡

Predspot has four main modules:

`dataset_preparation`: Module for preparing and managing crime datasets and study areas.

`crime_mapping`: Module for spatial and temporal crime mapping: point, hexagonal and square grids, KDE-based density surfaces and per-cell counts (`QuadratCount`).

`feature_engineering`: Module for time series feature engineering, including seasonality, trend, and difference features.

`ml_modelling`: Module that implements the prediction pipeline and model evaluation.

### Installation steps 🛠️

Predspot requires Python 3.10 or newer.

```bash
git clone https://github.com/adaj/predspot.git
cd predspot
pip install .
```

Core dependencies (installed automatically): pandas, geopandas, shapely,
numpy, scipy, scikit-learn, statsmodels and matplotlib. The optional
`geojsoncontour` package enables `predspot.utilities.contour_geojson`.

### Tests 🧪

```bash
pip install pytest
pytest
```

### Input Data Format 📊

The crime data should be a pandas DataFrame with the following required columns:
- `tag`: Crime type
- `t`: Timestamp
- `lon`: Longitude
- `lat`: Latitude

The study area should be a GeoDataFrame defining the geographical boundaries of interest.

## Resources 📚

For more information on the methods used in Predspot, please search more about these methods:
- Kernel Density Estimation for crime hotspot detection
- Time series decomposition for feature engineering
- Spatio-temporal crime prediction techniques

## License 📜

BSD 3-Clause

## Contributing 💡

Contributions are welcome! Please feel free to submit a Pull Request.

Guidelines for contributing:
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request


## Cite us

If you use Predspot in your research, please cite us:

APA:
```
Araujo, A., & Cacho, N. (2019). Predspot: Predicting crime hotspots with machine learning. Master’s thesis, UFRN (Universidade Federal do Rio Grande do Norte), Natal, Brazil.

Araújo, A., Cacho, N., Bezerra, L., Vieira, C., & Borges, J. (2018, June). Towards a crime hotspot detection framework for patrol planning. In 2018 IEEE 20th International Conference on High Performance Computing and Communications; IEEE 16th International Conference on Smart City; IEEE 4th International Conference on Data Science and Systems (HPCC/SmartCity/DSS) (pp. 1256-1263). IEEE.
```

or bibtex:
```
@article{araujo2019predspot,
  title={Predspot: Predicting crime hotspots with machine learning},
  author={Araujo, Adelson},
  year={2019},
  school={Universidade Federal do Rio Grande do Norte}
}

@inproceedings{araujo2018towards,
  title={Towards a crime hotspot detection framework for patrol planning},
  author={Ara{\'u}jo, Adelson and Cacho, N{\'a}dia and Bezerra, Lucas and Vieira, Carlos and Borges, Jo{\~a}o},
  booktitle={2018 IEEE 20th International Conference on High Performance Computing and Communications; IEEE 16th International Conference on Smart City; IEEE 4th International Conference on Data Science and Systems (HPCC/SmartCity/DSS)},
  pages={1256--1263},
  year={2018},
  organization={IEEE}
}
```


