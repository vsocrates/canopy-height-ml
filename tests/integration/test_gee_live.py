"""
Integration tests for query_gedi_earthengine and query_sentinel2 against real GEE.

These hit the actual Earth Engine API — run only when explicitly requested:
    uv run pytest -m integration -v

Requires:
    - GOOGLE_CLOUD_PROJECT env var (or .env)
    - GEE credentials at ~/.config/earthengine/credentials
      (run `uv run python -c "import ee; ee.Authenticate()"` once if missing)

AOI: small slice of Tahoe National Forest (~100km²) — minimal quota usage.
Date range: 30 days — enough shots to exercise the query without being slow.
"""

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

import ee

from canopy_height_prediction.agents import (
    GediQueryResult,
    IngestorDeps,
    Sentinel2QueryResult,
    query_gedi_earthengine,
    query_sentinel2,
)

# Small AOI: ~100 km² in Tahoe National Forest
SMALL_AOI = (-120.3, 39.3, -120.2, 39.4)
DATE_START = "2022-01-01"
DATE_END = "2023-01-01"  # Full year — 2022 has 15 orbit tracks over this AOI


class MockCtx:
    def __init__(self, deps):
        self.deps = deps


@pytest.fixture(scope="module", autouse=True)
def init_gee():
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "canopy-height-ml")
    ee.Initialize(project=project)


@pytest.mark.integration
def test_gedi_live_result_validates_against_model():
    deps = IngestorDeps(
        run_id="live-test",
        aoi_bbox=SMALL_AOI,
        date_start=DATE_START,
        date_end=DATE_END,
    )
    result = query_gedi_earthengine(MockCtx(deps))

    # Must parse cleanly into the Pydantic model
    validated = GediQueryResult(**result)

    assert validated.total_shots > 0, "Expected GEDI shots in Tahoe Jan-Mar 2023"
    assert validated.aoi_area_km2 > 0
    assert validated.raw_shot_density_per_km2 > 0
    # Histograms may have fewer than 10 bins if some ranges have no shots
    assert len(validated.rh98_histogram) <= 10
    assert len(validated.sensitivity_histogram) <= 10
    assert len(validated.slope_histogram) <= 10
    assert len(validated.rh98_histogram) > 0, "Expected at least one non-empty histogram bin"
    # GEE returns quality_flag histogram keys as float strings e.g. "0.0", "1.0"
    assert all(k.replace(".", "").lstrip("-").isdigit() for k in validated.quality_flag_counts)


@pytest.mark.integration
def test_gedi_live_histogram_bins_are_ascending():
    deps = IngestorDeps(
        run_id="live-test",
        aoi_bbox=SMALL_AOI,
        date_start=DATE_START,
        date_end=DATE_END,
    )
    result = query_gedi_earthengine(MockCtx(deps))

    bin_centers = [b[0] for b in result["rh98_histogram"]]
    assert bin_centers == sorted(bin_centers), "Histogram bins should be in ascending order"


@pytest.mark.integration
def test_gedi_live_density_consistent_with_shots_and_area():
    deps = IngestorDeps(
        run_id="live-test",
        aoi_bbox=SMALL_AOI,
        date_start=DATE_START,
        date_end=DATE_END,
    )
    result = query_gedi_earthengine(MockCtx(deps))

    expected = round(result["total_shots"] / result["aoi_area_km2"], 4)
    assert result["raw_shot_density_per_km2"] == pytest.approx(expected, rel=1e-3)


@pytest.mark.integration
def test_s2_live_result_validates_against_model():
    deps = IngestorDeps(
        run_id="live-test",
        aoi_bbox=SMALL_AOI,
        date_start=DATE_START,
        date_end=DATE_END,
    )
    result = query_sentinel2(MockCtx(deps))

    validated = Sentinel2QueryResult(**result)

    assert validated.scene_count >= 0
    assert 0.0 <= validated.mean_cloud_cover_pct <= 100.0
    # Expected S2 bands must all be present
    for band in ["B2", "B3", "B4", "B8", "B11", "B12", "SCL"]:
        assert band in validated.bands_available, f"Expected band {band} in bands_available"


@pytest.mark.integration
def test_s2_live_scene_count_plausible_for_date_range():
    """30-day window over a small AOI should yield at least a handful of S2 scenes."""
    deps = IngestorDeps(
        run_id="live-test",
        aoi_bbox=SMALL_AOI,
        date_start=DATE_START,
        date_end=DATE_END,
    )
    result = query_sentinel2(MockCtx(deps))

    # S2 revisit is ~5 days; 30-day window should give at least 2 scenes
    assert result["scene_count"] >= 2, (
        f"Expected >= 2 S2 scenes in a 30-day window, got {result['scene_count']}"
    )
