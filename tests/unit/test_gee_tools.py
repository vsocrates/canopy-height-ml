"""
Unit tests for query_gedi_earthengine and query_sentinel2.

Focused on: schema validation, derived field math, and edge cases.
API call structure (correct dataset IDs, filter args) is covered by integration tests.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from unittest.mock import MagicMock, patch

import pytest

from canopy_height_prediction.agents import (
    GediQueryResult,
    IngestorDeps,
    Sentinel2QueryResult,
    query_gedi_earthengine,
    query_sentinel2,
)

GEDI_PATCH = "canopy_height_prediction.agents._load_gedi_shots"

FAKE_HISTOGRAM = [[i * 6.0, 100 + i * 10] for i in range(10)]
FAKE_TOTAL_SHOTS = 1000
FAKE_AOI_AREA_KM2 = 10_000.0


class MockCtx:
    def __init__(self, deps):
        self.deps = deps


def make_deps(**overrides) -> IngestorDeps:
    defaults = dict(
        run_id="test-run-001",
        aoi_bbox=(-122.5, 37.5, -121.5, 38.5),
        date_start="2025-01-01",
        date_end="2025-03-01",
    )
    defaults.update(overrides)
    return IngestorDeps(**defaults)


def build_gedi_mocks(total_shots=FAKE_TOTAL_SHOTS, area_km2=FAKE_AOI_AREA_KM2):
    # Mock the merged collection returned by _load_gedi_shots
    mock_col = MagicMock()
    mock_col.size.return_value.getInfo.return_value = total_shots
    mock_col.reduceColumns.return_value.getInfo.return_value = {"histogram": FAKE_HISTOGRAM}
    mock_col.aggregate_histogram.return_value.getInfo.return_value = {"0": 200, "1": 800}

    # Mock ee.Geometry for the aoi.area() density calculation
    mock_ee = MagicMock()
    mock_aoi = MagicMock()
    mock_ee.Geometry.Rectangle.return_value = mock_aoi
    mock_aoi.area.return_value.divide.return_value.getInfo.return_value = area_km2

    return mock_ee, mock_col, mock_aoi


def build_s2_mocks(scene_count=45, cloud_mean=0.12):
    mock_ee = MagicMock()
    mock_ee.Geometry.Rectangle.return_value = MagicMock()

    mock_col = MagicMock()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value = mock_col
    mock_col.size.return_value.getInfo.return_value = scene_count
    mock_col.map.return_value.aggregate_stats.return_value.getInfo.return_value = {"mean": cloud_mean}
    return mock_ee


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_gedi_output_validates_against_model():
    mock_ee, mock_col, _ = build_gedi_mocks()
    with patch(GEDI_PATCH, return_value=mock_col), \
         patch("ee.Geometry", mock_ee.Geometry), \
         patch("ee.Reducer", mock_ee.Reducer):
        result = query_gedi_earthengine(MockCtx(make_deps()))

    validated = GediQueryResult(**result)
    assert validated.total_shots == FAKE_TOTAL_SHOTS
    assert len(validated.rh98_histogram) == 10
    assert len(validated.sensitivity_histogram) == 10
    assert len(validated.slope_histogram) == 10
    assert validated.quality_flag_counts == {"0": 200, "1": 800}


def test_s2_output_validates_against_model():
    mock_ee = build_s2_mocks(scene_count=45, cloud_mean=0.12)
    with patch("ee.ImageCollection", mock_ee.ImageCollection), \
         patch("ee.Geometry", mock_ee.Geometry), \
         patch("ee.Reducer", mock_ee.Reducer):
        result = query_sentinel2(MockCtx(make_deps()))

    validated = Sentinel2QueryResult(**result)
    assert validated.scene_count == 45
    assert validated.mean_cloud_cover_pct == 12.0
    assert {"B2", "B8", "B11", "B12", "SCL"}.issubset(set(validated.bands_available))


# ---------------------------------------------------------------------------
# Derived field math
# ---------------------------------------------------------------------------


def test_gedi_density_is_shots_divided_by_area():
    mock_ee, mock_col, _ = build_gedi_mocks(total_shots=500, area_km2=5_000.0)
    with patch(GEDI_PATCH, return_value=mock_col), \
         patch("ee.Geometry", mock_ee.Geometry), \
         patch("ee.Reducer", mock_ee.Reducer):
        result = query_gedi_earthengine(MockCtx(make_deps()))

    assert result["raw_shot_density_per_km2"] == pytest.approx(0.1, rel=1e-3)


def test_s2_cloud_cover_converted_from_fraction_to_percent():
    mock_ee = build_s2_mocks(cloud_mean=0.35)
    with patch("ee.ImageCollection", mock_ee.ImageCollection), \
         patch("ee.Geometry", mock_ee.Geometry), \
         patch("ee.Reducer", mock_ee.Reducer):
        result = query_sentinel2(MockCtx(make_deps()))

    assert result["mean_cloud_cover_pct"] == pytest.approx(35.0, abs=0.1)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_gedi_density_is_zero_when_area_is_zero():
    mock_ee, mock_col, _ = build_gedi_mocks(area_km2=0.0)
    with patch(GEDI_PATCH, return_value=mock_col), \
         patch("ee.Geometry", mock_ee.Geometry), \
         patch("ee.Reducer", mock_ee.Reducer):
        result = query_gedi_earthengine(MockCtx(make_deps()))

    assert result["raw_shot_density_per_km2"] == 0.0


def test_s2_cloud_cover_is_zero_when_gee_returns_null():
    mock_ee = build_s2_mocks()
    mock_ee.ImageCollection.return_value.filterBounds.return_value.filterDate.return_value \
        .map.return_value.aggregate_stats.return_value.getInfo.return_value = {"mean": None}
    with patch("ee.ImageCollection", mock_ee.ImageCollection), \
         patch("ee.Geometry", mock_ee.Geometry), \
         patch("ee.Reducer", mock_ee.Reducer):
        result = query_sentinel2(MockCtx(make_deps()))

    assert result["mean_cloud_cover_pct"] == 0.0
