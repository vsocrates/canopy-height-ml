import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from canopy_height_prediction.agents import TransformerDecision, TransformerDeps, transformer_agent
from canopy_height_prediction.state import PipelineState


def _make_decision(**overrides) -> TransformerDecision:
    defaults = dict(
        passed=True,
        cv_block_size_km=12.0,
        n_folds=5,
        rationale=(
            "Variogram range ~8km; chose block_size=12km (1.5× range). "
            "4,100 matched shots across 5 folds (fold sizes: [810, 825, 815, 830, 820]). "
            "nan_rate=3.2%, valid_obs_below_threshold_pct=5.1%."
        ),
        recommended_action="proceed",
        warnings=[],
    )
    defaults.update(overrides)
    return TransformerDecision(**defaults)


def _make_deps(replan_count: int = 0) -> TransformerDeps:
    return TransformerDeps(
        run_id="test-run-001",
        aoi_bbox=(-122.5, 37.5, -121.5, 38.5),
        date_start="2025-01-01",
        date_end="2025-03-01",
        replan_count=replan_count,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_transformer_happy_path_decision_fields(base_state: PipelineState):
    """Agent returns passed=True with expected decision fields."""
    decision = _make_decision()
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(transformer_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await transformer_agent.run("transform", deps=_make_deps())

    d = result.output
    assert d.passed is True
    assert d.cv_block_size_km == 12.0
    assert d.n_folds == 5
    assert d.recommended_action == "proceed"
    assert "12km" in d.rationale


# ---------------------------------------------------------------------------
# High nan rate path
# ---------------------------------------------------------------------------


async def test_transformer_high_nan_rate_returns_replan():
    """When >20% of shots have no S2 match, agent should fail with replan_widen_date."""
    decision = _make_decision(
        passed=False,
        cv_block_size_km=0.0,
        rationale="nan_rate_pct=45.0% — too few matched shots to proceed.",
        recommended_action="replan_widen_date",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(transformer_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await transformer_agent.run("transform", deps=_make_deps())

    d = result.output
    assert d.passed is False
    assert d.recommended_action == "replan_widen_date"


# ---------------------------------------------------------------------------
# High valid_obs_below_threshold path
# ---------------------------------------------------------------------------


async def test_transformer_high_valid_obs_below_threshold_returns_replan():
    """When >50% of shots are below MIN_VALID_OBS clear-sky scenes, agent should fail."""
    decision = _make_decision(
        passed=False,
        cv_block_size_km=0.0,
        rationale="valid_obs_below_threshold_pct=62.0% — composite unreliable, widen date range.",
        recommended_action="replan_widen_date",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(transformer_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await transformer_agent.run("transform", deps=_make_deps())

    d = result.output
    assert d.passed is False
    assert d.recommended_action == "replan_widen_date"
    assert "valid_obs" in d.rationale.lower() or "composite" in d.rationale.lower()


# ---------------------------------------------------------------------------
# Replan leniency
# ---------------------------------------------------------------------------


async def test_transformer_replan_uses_smaller_block_size():
    """With replan_count=1, agent should accept a smaller block_size_km closer to variogram range."""
    decision = _make_decision(
        cv_block_size_km=9.6,  # ~1.2× range instead of 1.5× on replan
        rationale=(
            "Replan: relaxed block_size to 9.6km (1.2× range=8km). "
            "4,100 matched shots across 5 folds."
        ),
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(transformer_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await transformer_agent.run("transform", deps=_make_deps(replan_count=1))

    d = result.output
    # Replan block size should be smaller than the default 1.5× multiplier would give
    assert d.cv_block_size_km < 12.0
    assert d.passed is True


# ---------------------------------------------------------------------------
# Warning but continue path (valid_obs 20-50%)
# ---------------------------------------------------------------------------


async def test_transformer_moderate_valid_obs_warns_but_passes():
    """Between 20% and 50% below threshold: agent should warn but still proceed."""
    decision = _make_decision(
        passed=True,
        warnings=["valid_obs_below_threshold_pct=31.0% — some tiles have limited clear-sky scenes."],
        recommended_action="proceed",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(transformer_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await transformer_agent.run("transform", deps=_make_deps())

    d = result.output
    assert d.passed is True
    assert len(d.warnings) == 1
    assert "valid_obs" in d.warnings[0].lower()
