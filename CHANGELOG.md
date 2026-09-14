# Changelog

All notable changes to Predspot are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-09

Revival release: the code base now targets Python 3.10+ with current
versions of pandas (>= 2.2), GeoPandas (>= 1.0), scikit-learn and statsmodels.

### Added
- `crime_mapping.load_study_area("City, Country")` fetches a study area
  boundary from OpenStreetMap via `osmnx` (`pip install predspot[osm]`).
- `synthetic.generate_crimes` generates synthetic events inside any study
  area: Gaussian hotspots plus uniform background, with trend, annual cycle,
  day-of-week and hour-of-day patterns; reproducible with `seed`.
- `QuadratCount` mapping (event counts per cell) as a first-class alternative
  to `KDE`, usable with hexagonal (`create_gridhexagonal`) and square
  (`create_gridsquares`) grids inside `PredictionPipeline`.
- `pipeline.build_default_pipeline` and reproducible
  `pipeline.generate_testdata(..., seed=...)`.
- `PredictionPipeline.features`, `.next_time` and `random_state`.
- Test suite (pytest) and continuous integration for Python 3.10-3.13.
- Documentation rebuilt with MkDocs (Material + mkdocstrings), deployed
  automatically to GitHub Pages; replaces the Sphinx site.
- `pyproject.toml` packaging (src layout) and automated PyPI publishing.

### Changed
- `tfreq` is optional in the feature classes (inferred from the series).
- Debug `print`s replaced with the `logging` module (`predspot` logger); the
  `debug=` arguments were removed.
- Wrapper estimators expose their inner estimator as `.estimator`
  (previously `._estimator`).
- `Dataset` no longer modifies the input DataFrame and requires the study
  area to have a CRS.
- Grid centroids are computed in a projected CRS; grids accept study areas in
  any CRS.
- `geojsoncontour` is an optional dependency (`pip install predspot[contour]`).

### Removed
- Sphinx documentation sources and the committed HTML build.
- `QuadratCount2`, `KGrid` and the hard dependencies on `descartes`,
  `contextily` and `rtree`.

### Fixed
- Compatibility with pandas 2/3 (`'ME'` offsets, `DataFrame.append`,
  positional `Series` indexing), GeoPandas 1.x (`sjoin(predicate=)`, CRS
  strings, `gpd.datasets`) and scikit-learn 1.x (`FeatureUnion` internals).
- `Seasonality`/`Trend` never called `STL(...).fit()`.
- `FeatureScaling` had no `fit`, so scalers inside a `Pipeline` were never fitted.

## [0.1.3] - 2020

Original master's thesis release.
