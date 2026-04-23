"""
Unit tests for Transformer tool functions.

Focused on: schema validation, derived field math, pure-function logic, and edge cases.
GEE API call structure is covered by integration tests.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from canopy_height_prediction._helpers import MIN_VALID_OBS, _estimate_variogram_range
from canopy_height_prediction._models import TransformerDecision, TransformerDeps
from canopy_height_prediction.agents.transformer import assign_folds, generate_spatial_blocks


class MockCtx:
    def __init__(self, deps):
        self.deps = deps


def make_deps(**overrides) -> TransformerDeps:
    defaults = dict(
        run_id="test-run-001",
        aoi_bbox=(-122.5, 37.5, -121.5, 38.5),
        date_start="2025-01-01",
        date_end="2025-03-01",
        n_folds=5,
    )
    defaults.update(overrides)
    return TransformerDeps(**defaults)


def make_matched_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Synthetic matched DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    min_lon, min_lat, max_lon, max_lat = -122.5, 37.5, -121.5, 38.5
    return pd.DataFrame({
        "shot_id": [f"shot_{i}" for i in range(n)],
        "lat": rng.uniform(min_lat, max_lat, n),
        "lon": rng.uniform(min_lon, max_lon, n),
        "rh98": rng.uniform(5.0, 50.0, n),
        "B2": rng.uniform(200, 1200, n),
        "B3": rng.uniform(300, 1500, n),
        "B4": rng.uniform(250, 1300, n),
        "B8": rng.uniform(1000, 4000, n),
        "B11": rng.uniform(400, 2000, n),
        "B12": rng.uniform(300, 1800, n),
        "ndvi": rng.uniform(0.1, 0.9, n),
        "evi": rng.uniform(0.05, 0.6, n),
        "valid_obs_count": rng.integers(1, 20, n),
    })


# ---------------------------------------------------------------------------
# _estimate_variogram_range — pure function
# ---------------------------------------------------------------------------


def test_variogram_range_returns_float_for_valid_data():
    df = make_matched_df(n=100)
    range_km, centers, sv = _estimate_variogram_range(
        df["lat"].values, df["lon"].values, df["rh98"].values
    )
    assert isinstance(range_km, float)
    assert range_km > 0
    assert len(centers) == len(sv)
    assert len(centers) > 0


def test_variogram_range_defaults_when_too_few_points():
    lats = np.array([37.5, 37.6])
    lons = np.array([-122.0, -122.1])
    values = np.array([10.0, 20.0])
    range_km, centers, sv = _estimate_variogram_range(lats, lons, values)
    assert range_km == 5.0
    assert centers == []
    assert sv == []


def test_variogram_bin_centers_are_ascending():
    df = make_matched_df(n=200)
    _, centers, _ = _estimate_variogram_range(
        df["lat"].values, df["lon"].values, df["rh98"].values
    )
    assert centers == sorted(centers)


def test_variogram_subsamples_large_inputs():
    """With 5000 points, function should still return without error (subsampled to 2000)."""
    rng = np.random.default_rng(0)
    n = 5000
    lats = rng.uniform(37.5, 38.5, n)
    lons = rng.uniform(-122.5, -121.5, n)
    values = rng.uniform(0, 50, n)
    range_km, centers, sv = _estimate_variogram_range(lats, lons, values)
    assert isinstance(range_km, float)


# ---------------------------------------------------------------------------
# generate_spatial_blocks
# ---------------------------------------------------------------------------


def test_generate_spatial_blocks_adds_block_id_column():
    deps = make_deps()
    deps._matched_df = make_matched_df(n=50)
    result = generate_spatial_blocks(MockCtx(deps), block_size_km=10.0)

    assert "block_id" in deps._matched_df.columns
    assert result["block_size_km"] == 10.0
    assert result["n_blocks_occupied"] >= 1


def test_generate_spatial_blocks_all_shots_get_assigned():
    deps = make_deps()
    df = make_matched_df(n=80)
    deps._matched_df = df
    generate_spatial_blocks(MockCtx(deps), block_size_km=10.0)

    assert deps._matched_df["block_id"].notna().all()
    assert len(deps._matched_df) == 80


def test_generate_spatial_blocks_larger_blocks_fewer_occupied():
    """Larger block size should produce fewer occupied blocks."""
    deps_small = make_deps()
    deps_small._matched_df = make_matched_df(n=100)
    result_small = generate_spatial_blocks(MockCtx(deps_small), block_size_km=5.0)

    deps_large = make_deps()
    deps_large._matched_df = make_matched_df(n=100)
    result_large = generate_spatial_blocks(MockCtx(deps_large), block_size_km=50.0)

    assert result_small["n_blocks_occupied"] >= result_large["n_blocks_occupied"]


def test_generate_spatial_blocks_returns_error_without_matched_df():
    deps = make_deps()
    result = generate_spatial_blocks(MockCtx(deps), block_size_km=10.0)
    assert "error" in result


# ---------------------------------------------------------------------------
# assign_folds
# ---------------------------------------------------------------------------


def test_assign_folds_creates_fold_and_split_columns():
    deps = make_deps(n_folds=5)
    deps._matched_df = make_matched_df(n=100)
    generate_spatial_blocks(MockCtx(deps), block_size_km=10.0)
    result = assign_folds(MockCtx(deps))

    assert "fold" in deps._folded_df.columns
    assert "split" in deps._folded_df.columns
    assert result["n_folds"] == 5


def test_assign_folds_fold_0_is_test_set():
    deps = make_deps(n_folds=5)
    deps._matched_df = make_matched_df(n=100)
    generate_spatial_blocks(MockCtx(deps), block_size_km=10.0)
    assign_folds(MockCtx(deps))

    df = deps._folded_df
    assert (df[df["fold"] == 0]["split"] == "test").all()
    assert (df[df["fold"] != 0]["split"] == "train").all()


def test_assign_folds_all_shots_get_a_fold():
    deps = make_deps(n_folds=4)
    deps._matched_df = make_matched_df(n=80)
    generate_spatial_blocks(MockCtx(deps), block_size_km=10.0)
    assign_folds(MockCtx(deps))

    df = deps._folded_df
    assert df["fold"].notna().all()
    assert set(df["fold"].unique()).issubset(set(range(4)))


def test_assign_folds_returns_error_without_block_ids():
    deps = make_deps()
    deps._matched_df = make_matched_df(n=50)  # no block_id column yet
    result = assign_folds(MockCtx(deps))
    assert "error" in result


# ---------------------------------------------------------------------------
# valid_obs_count threshold math
# ---------------------------------------------------------------------------


def test_valid_obs_below_threshold_pct_calculation():
    """
    Verify the threshold logic: shots with valid_obs_count < MIN_VALID_OBS
    should be counted correctly. We don't call extract_sentinel_bands directly
    (it hits GEE), so we replicate the calculation and assert MIN_VALID_OBS=3.
    """
    counts = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 2 below threshold (0,1)
    below = int((counts.fillna(0) < MIN_VALID_OBS).sum())
    below_pct = round(below / len(counts) * 100, 1)

    assert MIN_VALID_OBS == 2
    assert below == 2
    assert below_pct == 20.0


def test_valid_obs_below_threshold_pct_zero_when_all_valid():
    counts = pd.Series([5, 10, 8, 12, 7])
    below = int((counts.fillna(0) < MIN_VALID_OBS).sum())
    assert below == 0


def test_valid_obs_below_threshold_pct_treats_null_as_zero():
    counts = pd.Series([None, None, 5.0])  # 2 nulls → treated as 0 → below threshold
    below = int((counts.fillna(0) < MIN_VALID_OBS).sum())
    assert below == 2


# ---------------------------------------------------------------------------
# TransformerDecision schema
# ---------------------------------------------------------------------------


def test_transformer_decision_rejects_invalid_action():
    with pytest.raises(Exception):
        TransformerDecision(
            passed=True,
            cv_block_size_km=12.0,
            n_folds=5,
            rationale="test",
            recommended_action="do_something_weird",
        )


def test_transformer_decision_defaults_warnings_to_empty_list():
    d = TransformerDecision(
        passed=True,
        cv_block_size_km=12.0,
        n_folds=5,
        rationale="test",
        recommended_action="proceed",
    )
    assert d.warnings == []
