import os
import sys
import types

import geopandas as gpd
import pytest
from shapely.geometry import Point, box

from predspot.crime_mapping import load_study_area


def _fake_osmnx(monkeypatch, geometry):
    """Install a fake `osmnx` module whose geocode_to_gdf returns `geometry`."""
    calls = {}

    def geocode_to_gdf(query, which_result=None):
        calls["query"], calls["which_result"] = query, which_result
        return gpd.GeoDataFrame(
            {
                "osm_type": ["relation"],
                "osm_id": [1],
                "name": ["Natal"],
                "display_name": ["Natal, Rio Grande do Norte, Brasil"],
                "place_rank": [16],
            },
            geometry=[geometry],
            crs="EPSG:4326",
        )

    fake = types.ModuleType("osmnx")
    fake.geocode_to_gdf = geocode_to_gdf
    monkeypatch.setitem(sys.modules, "osmnx", fake)
    return calls


def test_load_study_area_returns_polygon(monkeypatch):
    calls = _fake_osmnx(monkeypatch, box(-35.3, -5.9, -35.2, -5.8))
    area = load_study_area("Natal, Brazil")
    assert calls == {"query": "Natal, Brazil", "which_result": None}
    assert list(area.columns) == ["name", "display_name", "osm_type", "osm_id", "geometry"]
    assert area.crs.to_epsg() == 4326
    assert area.geometry.iloc[0].geom_type == "Polygon"


def test_load_study_area_reprojects(monkeypatch):
    _fake_osmnx(monkeypatch, box(-35.3, -5.9, -35.2, -5.8))
    area = load_study_area("Natal, Brazil", crs="EPSG:31985")
    assert area.crs.to_epsg() == 31985


def test_load_study_area_rejects_points(monkeypatch):
    _fake_osmnx(monkeypatch, Point(-35.2, -5.8))
    with pytest.raises(ValueError, match="non-polygon"):
        load_study_area("Somewhere")


def test_load_study_area_without_osmnx(monkeypatch):
    monkeypatch.setitem(sys.modules, "osmnx", None)
    with pytest.raises(ImportError, match="predspot\\[osm\\]"):
        load_study_area("Natal, Brazil")


@pytest.mark.skipif(
    not os.environ.get("PREDSPOT_NETWORK_TESTS"),
    reason="set PREDSPOT_NETWORK_TESTS=1 to query OpenStreetMap",
)
def test_load_study_area_network():
    pytest.importorskip("osmnx")
    area = load_study_area("Natal, Rio Grande do Norte, Brazil")
    assert len(area) == 1
    assert area.geometry.iloc[0].geom_type in ("Polygon", "MultiPolygon")
    assert area.geometry.iloc[0].contains(Point(-35.2094, -5.7945))  # city centre
