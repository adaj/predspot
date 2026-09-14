# Predspot

**Predicting crime hotspots with machine learning.**

Predspot is a Python library for spatio-temporal crime prediction. It turns a
table of georeferenced, timestamped crime events into a grid of *places* and a
sequence of *periods*, builds time series features for every place and trains a
scikit-learn model to forecast where the next period's hotspots will be.

<figure markdown="span">
  ![Observed density and forecast](assets/forecast.png){ width="900" }
  <figcaption>Left: forecast density for the next month on a 500 m point grid.
  Right: observed monthly density and the forecast at the hottest grid point.</figcaption>
</figure>

## What it does

- **Spatio-temporal mapping** — kernel density estimation (KDE) on a grid of
  points, or event counts on hexagonal / square grids, at daily, weekly or
  monthly resolution.
- **Feature engineering** — lagged autoregressive, difference, seasonal and
  trend features (STL decomposition) for every place.
- **Prediction pipeline** — any scikit-learn regressor (or `Pipeline` with
  scaling and feature selection) trained to forecast the next period, with
  time series cross-validation.
- **Study areas from OpenStreetMap** — fetch a city boundary from a name.
- **Synthetic data** — generate realistic events (hotspots, trend, annual,
  weekly and hourly patterns) inside any study area to try things out.

## In a nutshell

```python
from predspot import Dataset, load_study_area, generate_crimes
from predspot.pipeline import build_default_pipeline

study_area = load_study_area("Natal, Rio Grande do Norte, Brazil")   # (1)!
crimes = generate_crimes(study_area, n_events=5000, seed=0)           # (2)!

dataset = Dataset(crimes, study_area)
pipeline = build_default_pipeline(study_area, tfreq="M", grid_resolution=1)
pipeline.fit(dataset)

print(pipeline.evaluate("r2", cv=3))   # time series cross-validation
forecast = pipeline.predict()          # density per grid point, next month
```

1. Needs `pip install "predspot[osm]"`. Any GeoDataFrame with a boundary works too.
2. Replace with your own DataFrame with `tag`, `t`, `lon`, `lat` columns.

Head to the [installation](getting-started/installation.md) and
[quickstart](getting-started/quickstart.md) pages, or read
[how Predspot works](guide/concepts.md).

## About

Predspot was created by [Adelson Araujo](https://github.com/adaj) as part of
his master's thesis at the Universidade Federal do Rio Grande do Norte (UFRN),
Brazil, and revived in 2026 for current versions of Python and its scientific
stack. It is research software released under the BSD-3-Clause license; see
[Citing Predspot](citing.md) if you use it in your work.
