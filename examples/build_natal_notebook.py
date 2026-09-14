"""Build examples/natal.ipynb from source cells.

The notebook itself is the deliverable; this script only exists so that the
notebook can be regenerated (and re-executed) reproducibly:

    python examples/build_natal_notebook.py
    jupyter nbconvert --to notebook --execute --inplace examples/natal.ipynb
"""

from pathlib import Path

import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Predspot end to end: crime hotspots for Natal, Brazil

This notebook walks through the whole Predspot workflow on the city of Natal
(Rio Grande do Norte, Brazil), using **synthetic** crime events so that it
runs anywhere without confidential police data:

1. fetch the city boundary from OpenStreetMap;
2. generate synthetic crime events with spatial hotspots and temporal patterns;
3. build the `Dataset`, the spatial grids and the spatio-temporal series (KDE and counts);
4. extract time series features;
5. fit, evaluate and inspect a `PredictionPipeline`;
6. forecast the next months and check how well the true hotspots are recovered.

Every intermediate object is displayed so you can see exactly what flows
between the steps. Requirements: `pip install "predspot[osm]" matplotlib`.
""")

code("""
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import predspot
from predspot import Dataset, PredictionPipeline, PandasFeatureUnion, generate_crimes, get_city_shape
from predspot.crime_mapping import KDE, QuadratCount, create_gridhexagonal, create_gridpoints
from predspot.feature_engineering import AR, Diff, Seasonality, Trend

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
plt.rcParams["figure.dpi"] = 100

print("predspot", predspot.__version__, "| geopandas", gpd.__version__, "| pandas", pd.__version__)
""")

md("""
## 1. Study area: the boundary of Natal

`get_city_shape` geocodes a place name with [osmnx](https://osmnx.readthedocs.io)
(Nominatim) and returns its boundary as a GeoDataFrame in WGS84. The query
should be specific enough for the first match to be the municipality.
""")

code("""
city = get_city_shape("Natal, RN, Brazil")
city
""")

code("""
polygon = city.geometry.iloc[0]
print("geometry:", polygon.geom_type, "| vertices:", len(polygon.exterior.coords))
print("bounds (W, S, E, N):", np.round(city.total_bounds, 4))
area_km2 = city.to_crs(city.estimate_utm_crs()).area.iloc[0] / 1e6
print(f"area: {area_km2:.1f} km²")

ax = city.plot(figsize=(5, 6), color="#f2f2f2", edgecolor="black")
ax.set_title("Natal, RN (OpenStreetMap boundary)")
ax.set_axis_off()
""")

md("""
## 2. Synthetic crime events

`generate_crimes` draws events from a space-time point process inside the
polygon: a mixture of Gaussian **hotspots** plus a uniform background in space,
and a temporal intensity with a linear **trend**, an **annual** cycle and
**day-of-week** / **hour-of-day** profiles. With `return_hotspots=True` we also
get the hotspot centres, which lets us check later whether the model finds them.
""")

code("""
crimes, hotspots = generate_crimes(
    city,
    n_events=12_000,
    start="2019-01-01",
    end="2021-12-31",
    n_hotspots=5,
    hotspot_share=0.65,
    hotspot_sd_km=0.7,
    trend=0.4,               # +40% events from start to end
    annual_amplitude=0.25,   # busier around the peak month...
    annual_peak_month=12,    # ...December
    seed=2019,
    return_hotspots=True,
)
crimes.head(10)
""")

code("""
print(crimes.shape)
crimes.describe(include="all").T
""")

code("""
hotspots
""")

code("""
crimes["tag"].value_counts().to_frame("events").assign(share=lambda d: (d["events"] / len(crimes)).round(3))
""")

md("""
### 2.1 Where and when do the events happen?
""")

code("""
fig, ax = plt.subplots(figsize=(7, 8))
city.plot(ax=ax, color="#f7f7f7", edgecolor="black")
sample = crimes.sample(4000, random_state=0)
ax.scatter(sample["lon"], sample["lat"], s=3, alpha=0.35, color="#1f4e79", label="events (sample)")
hotspots.plot(ax=ax, color="#d62728", marker="*", markersize=150, zorder=5, label="hotspot centres")
ax.legend(loc="lower left")
ax.set_title("Synthetic crime events in Natal")
ax.set_axis_off()
""")

code("""
monthly = crimes.set_index("t").resample("ME").size().rename("events")
fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
monthly.plot(ax=axes[0], marker="o")
axes[0].set_title("Events per month (trend + annual cycle)")
axes[0].set_xlabel("")
crimes["t"].dt.day_name().value_counts().reindex(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
).plot.bar(ax=axes[1], color="#1f4e79")
axes[1].set_title("Events per weekday")
crimes["t"].dt.hour.value_counts().sort_index().plot.bar(ax=axes[2], color="#1f4e79", width=0.9)
axes[2].set_title("Events per hour of day")
fig.tight_layout()
""")

md("""
## 3. Dataset, grids and the spatio-temporal series

`Dataset` validates the events (columns `tag`, `t`, `lon`, `lat`) and turns
them into a GeoDataFrame of points in WGS84, together with the study area.
""")

code("""
dataset = Dataset(crimes, city)
dataset
""")

code("""
dataset.crimes.head()
""")

md("""
### 3.1 Grids

Predspot discretises the city into **places**. Two kinds of grid are built
below: a grid of **points** every 500 m (used by `KDE`) and a grid of
**hexagons** of 1 km² (used by `QuadratCount`). Only cells intersecting the
city are kept; each grid has `lon`/`lat` columns with the cell centroid and an
index named `places`.
""")

code("""
points = create_gridpoints(city, resolution=0.5)
hexes = create_gridhexagonal(city, resolution=1.0)
print(f"{len(points)} grid points at 500 m | {len(hexes)} hexagons of 1 km²")
points.head()
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 7))
city.boundary.plot(ax=axes[0], color="black")
points.plot(ax=axes[0], markersize=4, color="#1f4e79")
axes[0].set_title(f"Point grid, 500 m ({len(points)} places)")
city.boundary.plot(ax=axes[1], color="black")
hexes.plot(ax=axes[1], facecolor="none", edgecolor="#1f4e79", linewidth=0.6)
axes[1].set_title(f"Hexagonal grid, 1 km² ({len(hexes)} places)")
for ax in axes:
    ax.set_axis_off()
""")

md("""
### 3.2 Spatio-temporal mapping

A mapping assigns one value to every `(period, place)` pair. `KDE` fits a
Gaussian kernel density estimate to the events of each period and evaluates it
at the grid points; `QuadratCount` counts the events inside each polygon. Both
return the same object: a `pandas.Series` named `crime_density` with a
`(t, places)` MultiIndex — the **spatio-temporal series**.
""")

code("""
kde = KDE(tfreq="M", grid=points, bandwidth="silverman")
stseries = kde.fit_transform(dataset.crimes)
print(type(stseries).__name__, stseries.shape, "| KDE factor:", round(kde.factor, 4))
stseries.head(8)
""")

code("""
# The same series in wide form: one row per month, one column per place
wide = stseries.unstack("places")
wide.iloc[:6, :8]
""")

code("""
counts = QuadratCount(tfreq="M", grid=hexes).fit_transform(dataset.crimes)
counts_wide = counts.unstack("places")
print("events per month recovered by the counts:", counts_wide.sum(axis=1).astype(int).head(3).tolist(), "...")
counts_wide.iloc[:6, :8]
""")

code("""
month = pd.Timestamp("2021-06-30")
fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

g = points.copy()
g["density"] = stseries.xs(month, level="t").reindex(points.index).values
g.plot(ax=axes[0], column="density", cmap="viridis", markersize=12, marker="s", legend=True,
       legend_kwds={"shrink": 0.6, "label": "KDE density"})
city.boundary.plot(ax=axes[0], color="black", linewidth=1)
hotspots.plot(ax=axes[0], color="#d62728", marker="*", markersize=150, zorder=5)
axes[0].set_title(f"KDE on the point grid - {month:%B %Y}")

h = hexes.copy()
h["events"] = counts.xs(month, level="t").reindex(hexes.index).values
h.plot(ax=axes[1], column="events", cmap="viridis", edgecolor="white", linewidth=0.3, legend=True,
       legend_kwds={"shrink": 0.6, "label": "events in the cell"})
city.boundary.plot(ax=axes[1], color="black", linewidth=1)
hotspots.plot(ax=axes[1], color="#d62728", marker="*", markersize=150, zorder=5)
axes[1].set_title(f"QuadratCount on hexagons - {month:%B %Y}")
for ax in axes:
    ax.set_axis_off()
""")

md("""
### 3.3 The KDE bandwidth

`bandwidth="silverman"` estimates the kernel width from the spread of the
events in the first period and keeps it fixed, which makes densities comparable
over time but tends to over-smooth a whole city. A numeric `bandwidth` is used
directly as the KDE factor (a multiple of the city-wide spread of the events):
smaller values give sharper maps. Compare three settings for the same month —
this is the trade-off discussed in the thesis (Chapter 2, Figure 3).
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
for ax, bw in zip(axes, ["silverman", 0.2, 0.08]):
    m = KDE(tfreq="M", grid=points, bandwidth=bw)
    st = m.fit_transform(dataset.crimes)
    g = points.copy()
    g["density"] = st.xs(month, level="t").reindex(points.index).values
    g.plot(ax=ax, column="density", cmap="viridis", markersize=10, marker="s")
    city.boundary.plot(ax=ax, color="black", linewidth=1)
    hotspots.plot(ax=ax, color="#d62728", marker="*", markersize=120, zorder=5)
    ax.set_title(f"bandwidth={bw!r} (factor {m.factor:.3f})")
    ax.set_axis_off()
""")

md("""
## 4. Time series features

From here on we use `bandwidth=0.2`, a middle ground between the
over-smoothed Silverman estimate and a noisy small kernel.

Each place has a monthly series. The feature classes transform it (STL trend,
STL seasonal component, first difference or the raw series) and build `lags`
lagged columns. `PandasFeatureUnion` aligns them on the `(t, places)` index and
drops the warm-up rows. Note the extra row for the month **after** the last
observed one — that is the row the model will forecast.
""")

code("""
BANDWIDTH = 0.2
stseries = KDE(tfreq="M", grid=points, bandwidth=BANDWIDTH).fit_transform(dataset.crimes)

LAGS = 6
fextraction = PandasFeatureUnion([
    ("ar", AR(lags=3)),
    ("seasonal", Seasonality(lags=LAGS)),
    ("trend", Trend(lags=LAGS)),
    ("diff", Diff(lags=LAGS)),
])
X = fextraction.fit_transform(stseries)
print(X.shape, "| periods:", X.index.get_level_values("t").min().date(), "->", X.index.get_level_values("t").max().date())
X.head(8)
""")

code("""
# What the decomposition looks like for the busiest place
busiest = stseries.groupby("places").mean().idxmax()
ts = stseries.xs(busiest, level="places")
from statsmodels.tsa.seasonal import STL
res = STL(ts, period=LAGS).fit()
fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
ts.plot(ax=axes[0], title=f"place {busiest}: KDE density per month")
res.trend.plot(ax=axes[1], title="STL trend")
res.seasonal.plot(ax=axes[2], title=f"STL seasonal (period={LAGS})")
ts.diff().plot(ax=axes[3], title="first difference")
for ax in axes:
    ax.set_xlabel("")
fig.tight_layout()
""")

md("""
## 5. Fit and evaluate a prediction pipeline

The pipeline chains the mapping, the feature extraction and a scikit-learn
estimator. Here the estimator is itself a `Pipeline`: quantile scaling,
recursive feature elimination and a random forest, each wrapped so that
DataFrames (and the `(t, places)` index) survive every step.
""")

code("""
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

from predspot.feature_engineering import FeatureScaling
from predspot.ml_modelling import FeatureSelection, Model

pipeline = PredictionPipeline(
    mapping=KDE(tfreq="M", grid=points, bandwidth=BANDWIDTH),
    fextraction=PandasFeatureUnion([
        ("ar", AR(lags=3)),
        ("seasonal", Seasonality(lags=LAGS)),
        ("trend", Trend(lags=LAGS)),
        ("diff", Diff(lags=LAGS)),
    ]),
    estimator=Pipeline([
        ("scaling", FeatureScaling(QuantileTransformer(n_quantiles=20, output_distribution="uniform"))),
        ("selection", FeatureSelection(RFE(RandomForestRegressor(n_estimators=20, random_state=0, n_jobs=-1),
                                           n_features_to_select=12, step=3))),
        ("model", Model(RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=-1))),
    ]),
    random_state=0,
)
pipeline.fit(dataset)
print("training rows:", pipeline.features.loc[: stseries.index.get_level_values("t").max()].shape[0])
print("next period to forecast:", pipeline.next_time.date())
""")

code("""
scores = pipeline.evaluate(["r2", "mse"], cv=3)   # time series CV: train on earlier folds, test on the next
scores.loc["mean"] = scores.mean()
scores
""")

code("""
fi = pipeline.feature_importances
ax = fi.sort_values("importance").plot.barh(figsize=(7, 5), legend=False, color="#1f4e79")
ax.set_title("Selected features and their importance")
fi.head(12)
""")

md("""
## 6. Forecast the next months

`predict()` returns the density of every place for the period after the last
observed one. Each call appends its forecast to the series, recomputes the
features and moves the horizon one period forward, so calling it three times
gives a three-month recursive forecast.
""")

code("""
forecasts = pd.concat([pipeline.predict() for _ in range(3)])
forecasts.groupby("t").describe().round(2)
""")

code("""
first_month = forecasts.index.get_level_values("t").min()
fc = points.join(forecasts.xs(first_month, level="t"))
top10 = fc.nlargest(10, "crime_density")[["lon", "lat", "crime_density"]].round(4)
print(f"Top-10 places for {first_month:%B %Y}:")
top10
""")

code("""
fig, ax = plt.subplots(figsize=(7, 8))
fc.plot(ax=ax, column="crime_density", cmap="magma", markersize=14, marker="s", legend=True,
        legend_kwds={"shrink": 0.6, "label": "forecast density"})
city.boundary.plot(ax=ax, color="black", linewidth=1)
hotspots.plot(ax=ax, color="#00e5ff", marker="*", markersize=170, zorder=5, label="true hotspot centres")
top = fc.nlargest(20, "crime_density")
ax.scatter(top["lon"], top["lat"], s=110, facecolors="none", edgecolors="#00e5ff", linewidths=1.6, label="top-20 forecast")
ax.legend(loc="lower left")
ax.set_title(f"Forecast hotspots for {first_month:%B %Y}")
ax.set_axis_off()
""")

md("""
### 6.1 Does the forecast find the true hotspots?

Since the data are synthetic we know where the hotspots are. For each true
hotspot centre we look up the nearest grid point and check how it ranks in the
forecast (1 = hottest place of the city). Hotspots holding a larger share of the
events should rank near the top; small hotspots on a 500 m grid compete with the
many grid points that surround the biggest one.
""")

code("""
def distance_km(lon1, lat1, lon2, lat2):
    dx = (lon1 - lon2) * 111.32 * np.cos(np.radians((lat1 + lat2) / 2))
    dy = (lat1 - lat2) * 110.57
    return np.hypot(dx, dy)

ranked = fc.sort_values("crime_density", ascending=False)
ranked["rank"] = np.arange(1, len(ranked) + 1)
rows = []
for i, h in hotspots.iterrows():
    d = distance_km(ranked["lon"].values, ranked["lat"].values, h.geometry.x, h.geometry.y)
    nearest = ranked.iloc[int(d.argmin())]
    rows.append({
        "hotspot": i,
        "share_of_events": round(h["share"], 3),
        "nearest_place": int(nearest.name),
        "distance_km": round(float(d.min()), 2),
        "forecast_rank": int(nearest["rank"]),
        "percentile": round(100 * (1 - nearest["rank"] / len(ranked)), 1),
    })
pd.DataFrame(rows).set_index("hotspot").sort_values("forecast_rank")
""")

code("""
# The full history + forecast of the top forecast place
place = fc["crime_density"].idxmax()
series = pipeline.stseries.xs(place, level="places")
observed = series.loc[: stseries.index.get_level_values("t").max()]
predicted = series.loc[forecasts.index.get_level_values("t").min():]
ax = observed.plot(figsize=(10, 3.5), marker="o", label="observed (KDE)")
predicted.plot(ax=ax, marker="*", markersize=12, linestyle="--", color="#d62728", label="forecast")
ax.set_title(f"Place {place}: monthly density and 3-month recursive forecast")
ax.set_xlabel("")
ax.legend()
""")

md("""
## 7. The same pipeline with counts on hexagons

`QuadratCount` is a drop-in replacement for `KDE`: swap the mapping and the
grid, keep everything else.
""")

code("""
hex_pipeline = PredictionPipeline(
    mapping=QuadratCount(tfreq="M", grid=hexes),
    fextraction=PandasFeatureUnion([
        ("ar", AR(lags=3)),
        ("seasonal", Seasonality(lags=LAGS)),
        ("trend", Trend(lags=LAGS)),
    ]),
    estimator=RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=0, n_jobs=-1),
    random_state=0,
).fit(dataset)
print("r2 per fold:", np.round(hex_pipeline.evaluate("r2", cv=3), 3))
hex_fc = hexes.join(hex_pipeline.predict().droplevel("t"))

fig, ax = plt.subplots(figsize=(7, 8))
hex_fc.plot(ax=ax, column="crime_density", cmap="magma", edgecolor="white", linewidth=0.3, legend=True,
            legend_kwds={"shrink": 0.6, "label": "forecast events"})
city.boundary.plot(ax=ax, color="black", linewidth=1)
hotspots.plot(ax=ax, color="#00e5ff", marker="*", markersize=170, zorder=5)
ax.set_title(f"Forecast events per hexagon for {hex_pipeline.stseries.index.get_level_values('t').max():%B %Y}")
ax.set_axis_off()
""")

md("""
## Wrapping up

- `get_city_shape` gave us the study area; `generate_crimes` produced events with known hotspots.
- `Dataset` → grid → `KDE` / `QuadratCount` turned the events into a spatio-temporal series.
- Lagged STL trend/seasonal, difference and autoregressive features fed a scikit-learn pipeline
  wrapped by `PredictionPipeline`, evaluated with time series cross-validation.
- `predict()` forecast the following months; the grid points nearest to the true hotspot
  centres rank at the top of the forecast, in proportion to each hotspot's share of events.

To run this on real data, replace the synthetic `crimes` DataFrame with your own
`tag, t, lon, lat` table and model each crime type (`tag`) separately, as the
framework recommends. See the [documentation](https://adaj.github.io/predspot/)
for the details of every step.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).with_name("natal.ipynb")
nbf.write(nb, out)
print(f"wrote {out} with {len(cells)} cells")
