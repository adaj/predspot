# Feature engineering

The spatio-temporal series is a panel: one time series per place. Feature
classes in [`predspot.feature_engineering`][predspot.feature_engineering]
turn each of those series into lagged predictors.

## Feature classes

Every class takes `lags` (how many previous periods become columns) and an
optional `tfreq` (inferred from the series when omitted):

| Class | Transformation before lagging | Columns |
|-------|-------------------------------|---------|
| [`AR`][predspot.feature_engineering.AR] | none (raw values) | `ar_1 … ar_k` |
| [`Diff`][predspot.feature_engineering.Diff] | first difference | `diff_1 … diff_k` |
| [`Seasonality`][predspot.feature_engineering.Seasonality] | seasonal component of an STL decomposition with period `lags` | `seasonal_1 … seasonal_k` |
| [`Trend`][predspot.feature_engineering.Trend] | trend component of the same STL decomposition | `trend_1 … trend_k` |

`ar_1` at period `t` is the value at `t-1`, `ar_2` the value at `t-2`, and so
on. For `Seasonality` and `Trend`, `lags` doubles as the STL period: use 12
for monthly data with a yearly cycle, 7 for daily data with a weekly cycle,
52 for weekly data, and make sure the series is at least twice as long.

```python
from predspot.feature_engineering import AR, Seasonality

X = AR(lags=3).fit_transform(stseries)
X.head()
```

```
                    ar_1   ar_2   ar_3
t          places
2019-04-30 0       14.83  14.02  12.31
           1       ...
```

Note the index: features at `t` only use values **before** `t`, and the
matrix always includes one extra period after the last observed one — the
row a fitted model uses to forecast.

## Combining features

[`PandasFeatureUnion`][predspot.utilities.PandasFeatureUnion] runs several
feature transformers and concatenates their columns, aligning on the
`(t, places)` index and dropping rows that any transformer could not fill
(warm-up periods):

```python
from predspot import PandasFeatureUnion
from predspot.feature_engineering import Seasonality, Trend, Diff

fextraction = PandasFeatureUnion([
    ("seasonal", Seasonality(lags=12)),
    ("trend", Trend(lags=12)),
    ("diff", Diff(lags=12)),
])
X = fextraction.fit_transform(stseries)
```

## Scaling and selection inside the estimator

Because features are a DataFrame, plain scikit-learn transformers would drop
the index. [`FeatureScaling`][predspot.feature_engineering.FeatureScaling]
and [`FeatureSelection`][predspot.ml_modelling.FeatureSelection] wrap any
scaler / selector so that DataFrames come out the other side, and
[`Model`][predspot.ml_modelling.Model] does the same for the regressor:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from predspot.feature_engineering import FeatureScaling
from predspot.ml_modelling import FeatureSelection, Model

estimator = Pipeline([
    ("scaling", FeatureScaling(QuantileTransformer(n_quantiles=10))),
    ("selection", FeatureSelection(RFE(RandomForestRegressor(n_estimators=20)))),
    ("model", Model(RandomForestRegressor(n_estimators=100))),
])
```

## Writing your own feature

Subclass [`TimeSeriesFeatures`][predspot.feature_engineering.TimeSeriesFeatures],
set `label` and implement `apply_ts_decomposition(ts)`, which receives the
series of one place and returns the series to lag:

```python
from predspot.feature_engineering import TimeSeriesFeatures

class RollingMean(TimeSeriesFeatures):
    @property
    def label(self):
        return "rmean"

    def apply_ts_decomposition(self, ts):
        return ts.rolling(3, min_periods=1).mean()
```
