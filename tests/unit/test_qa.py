"""
Unit tests for the QA agent and its tools.

Focused on: tool logic (flagging thresholds, stats math), schema validation,
and agent-level decision shapes. No DB or model calls made.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from canopy_height_prediction.agents import (
    QADecision,
    QADeps,
    check_feature_distributions,
    check_fold_balance,
    check_target_range,
    qa_agent,
)


class MockCtx:
    def __init__(self, deps):
        self.deps = deps


def make_deps(**overrides) -> QADeps:
    defaults = dict(run_id="test-run-001")
    defaults.update(overrides)
    return QADeps(**defaults)


def make_cleaned_df(
    n: int = 100,
    rh98_range=(5.0, 45.0),
    nan_bands: list[str] | None = None,
    n_folds: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "shot_id": [f"shot_{i}" for i in range(n)],
        "lat": rng.uniform(37.5, 38.5, n),
        "lon": rng.uniform(-122.5, -121.5, n),
        "rh98": rng.uniform(*rh98_range, n),
        "b2": rng.uniform(200, 1200, n),
        "b3": rng.uniform(300, 1500, n),
        "b4": rng.uniform(250, 1300, n),
        "b8": rng.uniform(1000, 4000, n),
        "b11": rng.uniform(400, 2000, n),
        "b12": rng.uniform(300, 1800, n),
        "ndvi": rng.uniform(0.1, 0.9, n),
        "evi": rng.uniform(0.05, 0.6, n),
        "scl_valid_obs": rng.integers(3, 20, n),
        "fold": [i % n_folds for i in range(n)],
        "split": ["test" if i % n_folds == 0 else "train" for i in range(n)],
    })
    if nan_bands:
        for col in nan_bands:
            df.loc[rng.choice(n, size=int(n * 0.15), replace=False), col] = None
    return df


def _make_qa_decision(**overrides) -> QADecision:
    defaults = dict(
        passed=True,
        issues=[],
        rationale="rh98 range [5.0, 45.0]m — 0% negative. Fold sizes balanced. No issues.",
        recommended_action="proceed",
    )
    defaults.update(overrides)
    return QADecision(**defaults)


# ---------------------------------------------------------------------------
# check_feature_distributions
# ---------------------------------------------------------------------------


def test_check_feature_distributions_no_flags_on_clean_data():
    deps = make_deps()
    deps._cleaned_df = make_cleaned_df()
    result = check_feature_distributions(MockCtx(deps))

    assert "error" not in result
    assert result["flagged_bands"] == []
    assert "b8" in result["band_stats"]


def test_check_feature_distributions_flags_high_nan_band():
    deps = make_deps()
    deps._cleaned_df = make_cleaned_df(nan_bands=["b11"])
    result = check_feature_distributions(MockCtx(deps))

    flagged = result["flagged_bands"]
    assert any("b11" in f for f in flagged), f"Expected b11 flagged, got: {flagged}"


def test_check_feature_distributions_returns_error_without_df():
    deps = make_deps()
    result = check_feature_distributions(MockCtx(deps))
    assert "error" in result


def test_check_feature_distributions_stats_are_correct():
    deps = make_deps()
    df = make_cleaned_df()
    deps._cleaned_df = df
    result = check_feature_distributions(MockCtx(deps))

    b8_stats = result["band_stats"]["b8"]
    assert b8_stats["mean"] == pytest.approx(df["b8"].mean(), rel=1e-2)
    assert b8_stats["nan_rate_pct"] == 0.0


# ---------------------------------------------------------------------------
# check_fold_balance
# ---------------------------------------------------------------------------


def test_check_fold_balance_no_flags_on_balanced_folds():
    deps = make_deps()
    deps._cleaned_df = make_cleaned_df(n=100, n_folds=5)
    result = check_fold_balance(MockCtx(deps))

    assert result["flagged_folds"] == []
    assert result["n_total"] == 100
    assert len(result["fold_sizes"]) == 5


def test_check_fold_balance_flags_dominant_fold():
    deps = make_deps()
    # Manually create severely imbalanced folds: fold 0 = 60%, others split the rest
    n = 100
    folds = [0] * 60 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10
    df = make_cleaned_df(n=n)
    df["fold"] = folds
    deps._cleaned_df = df
    result = check_fold_balance(MockCtx(deps))

    assert any("fold 0" in f for f in result["flagged_folds"])


def test_check_fold_balance_flags_tiny_fold():
    deps = make_deps()
    n = 100
    folds = [0] * 2 + [1] * 25 + [2] * 25 + [3] * 24 + [4] * 24
    df = make_cleaned_df(n=n)
    df["fold"] = folds
    deps._cleaned_df = df
    result = check_fold_balance(MockCtx(deps))

    assert any("fold 0" in f for f in result["flagged_folds"])


def test_check_fold_balance_returns_error_without_df():
    deps = make_deps()
    result = check_fold_balance(MockCtx(deps))
    assert "error" in result


# ---------------------------------------------------------------------------
# check_target_range
# ---------------------------------------------------------------------------


def test_check_target_range_no_flags_on_valid_rh98():
    deps = make_deps()
    deps._cleaned_df = make_cleaned_df(rh98_range=(2.0, 55.0))
    result = check_target_range(MockCtx(deps))

    assert result["flagged"] == []
    assert result["rh98_min"] >= 2.0
    assert result["rh98_max"] <= 55.0


def test_check_target_range_flags_negative_values():
    deps = make_deps()
    df = make_cleaned_df(n=100, rh98_range=(5.0, 40.0))
    # Force 10 negative values (10% > 5% threshold)
    df.loc[:9, "rh98"] = -1.0
    deps._cleaned_df = df
    result = check_target_range(MockCtx(deps))

    assert any("<= 0" in f for f in result["flagged"])
    assert result["negative_pct"] == pytest.approx(10.0, abs=0.1)


def test_check_target_range_flags_implausibly_high_values():
    deps = make_deps()
    df = make_cleaned_df(n=100, rh98_range=(5.0, 40.0))
    # Force 15 values > 60m (15% > 10% threshold)
    df.loc[:14, "rh98"] = 75.0
    deps._cleaned_df = df
    result = check_target_range(MockCtx(deps))

    assert any("60m" in f for f in result["flagged"])
    assert result["implausible_high_pct"] == pytest.approx(15.0, abs=0.1)


def test_check_target_range_flags_null_rh98():
    deps = make_deps()
    df = make_cleaned_df(n=50)
    df.loc[:4, "rh98"] = None
    deps._cleaned_df = df
    result = check_target_range(MockCtx(deps))

    assert result["n_null_rh98"] == 5
    assert any("null rh98" in f for f in result["flagged"])


def test_check_target_range_returns_error_without_df():
    deps = make_deps()
    result = check_target_range(MockCtx(deps))
    assert "error" in result


# ---------------------------------------------------------------------------
# QA agent — decision shapes
# ---------------------------------------------------------------------------


async def test_qa_happy_path_passes():
    decision = _make_qa_decision()
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(qa_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await qa_agent.run("validate", deps=make_deps())

    d = result.output
    assert d.passed is True
    assert d.recommended_action == "proceed"
    assert d.issues == []


async def test_qa_empty_dataset_returns_replan():
    decision = _make_qa_decision(
        passed=False,
        issues=["0 rows in gedi_shots_cleaned — no data to validate"],
        rationale="No cleaned shots found for this run.",
        recommended_action="replan_widen_date",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(qa_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await qa_agent.run("validate", deps=make_deps())

    d = result.output
    assert d.passed is False
    assert d.recommended_action == "replan_widen_date"


async def test_qa_bad_target_range_returns_replan():
    decision = _make_qa_decision(
        passed=False,
        issues=["12.0% of rh98 values are <= 0 (physically impossible)"],
        rationale="12% negative rh98 values — likely over-filtering removed valid shots.",
        recommended_action="replan_relax_thresholds",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(qa_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await qa_agent.run("validate", deps=make_deps())

    d = result.output
    assert d.passed is False
    assert d.recommended_action == "replan_relax_thresholds"
    assert len(d.issues) > 0


async def test_qa_corrupt_data_aborts():
    decision = _make_qa_decision(
        passed=False,
        issues=["All band values are null — S2 sampling failed completely"],
        rationale="100% NaN across all S2 bands — dataset is unusable.",
        recommended_action="abort",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(qa_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await qa_agent.run("validate", deps=make_deps())

    assert result.output.recommended_action == "abort"


# ---------------------------------------------------------------------------
# QADecision schema
# ---------------------------------------------------------------------------


def test_qa_decision_rejects_invalid_action():
    with pytest.raises(Exception):
        QADecision(
            passed=True,
            rationale="test",
            recommended_action="invalid_action",
        )


def test_qa_decision_defaults_issues_to_empty_list():
    d = QADecision(passed=True, rationale="test", recommended_action="proceed")
    assert d.issues == []
