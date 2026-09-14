# Installation

Predspot requires **Python 3.10 or newer** and is published on PyPI:

```bash
pip install predspot
```

Optional extras add features that need heavier dependencies:

| Extra | Installs | Enables |
|-------|----------|---------|
| `osm` | [osmnx](https://osmnx.readthedocs.io) | [`load_study_area`][predspot.crime_mapping.load_study_area] — study areas from OpenStreetMap |
| `contour` | [geojsoncontour](https://github.com/bartromgens/geojsoncontour) | [`contour_geojson`][predspot.utilities.contour_geojson] — GeoJSON contour export |
| `dev` | pytest, ruff, build, twine | Running the test suite and building the package |

```bash
pip install "predspot[osm,contour]"
```

The core dependencies — pandas, GeoPandas, Shapely, NumPy, SciPy, scikit-learn,
statsmodels and Matplotlib — are installed automatically.

!!! tip "conda users"
    GeoPandas and its GEOS/PROJ stack install fine from PyPI wheels nowadays,
    but if you prefer conda: `conda install -c conda-forge geopandas` first,
    then `pip install predspot` in the same environment.

## From source

```bash
git clone https://github.com/adaj/predspot.git
cd predspot
pip install -e ".[dev,osm,contour]"
pytest
```

See [Contributing](../contributing.md) for the development workflow.

## Checking the installation

```python
import predspot
print(predspot.__version__)
```
