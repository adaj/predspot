import json

import pytest

from predspot import crime_mapping as cm
from predspot.utilities import contour_geojson


def test_contour_geojson(dataset, study_area):
    pytest.importorskip('geojsoncontour')
    grid = cm.create_gridpoints(study_area, resolution=1)
    st = cm.KDE(tfreq='M', grid=grid).fit_transform(dataset.crimes)
    month = st.xs(st.index.get_level_values('t')[0], level='t')
    geojson = contour_geojson(month, study_area, 1, cmin=0, cmax=month.max())
    assert json.loads(geojson)['type'] == 'FeatureCollection'
