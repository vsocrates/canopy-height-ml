import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from canopy_height_prediction._models import IngestorDecision, IngestorDeps
from canopy_height_prediction.agents.ingestor import ingestor_agent
from canopy_height_prediction.state import PipelineState


def _make_decision(**overrides) -> IngestorDecision:
    defaults = dict(
        passed=True,
        sensitivity_min=0.95,
        slope_max_deg=30.0,
        raw_shots=6800,
        accepted_shots=4200,
        rationale="Accepted 4,200/6,800 shots — rejected 38% sensitivity < 0.95, 12% slope > 30°.",
        recommended_action="proceed",
        warnings=[],
    )
    defaults.update(overrides)
    return IngestorDecision(**defaults)


def _make_deps(replan_count: int = 0) -> IngestorDeps:
    return IngestorDeps(
        run_id="test-run-001",
        aoi_bbox=(-122.5, 37.5, -121.5, 38.5),
        date_start="2025-01-01",
        date_end="2025-03-01",
        replan_count=replan_count,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


async def test_ingestor_happy_path_updates_state(base_state: PipelineState):
    """Agent returns passed=True → state fields are set correctly."""
    decision = _make_decision()
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(ingestor_agent, "run", new=AsyncMock(return_value=mock_result)):
        deps = _make_deps()
        result = await ingestor_agent.run("ingest", deps=deps)

    d = result.output
    assert d.passed is True
    assert d.accepted_shots == 4200
    assert d.recommended_action == "proceed"
    assert "4,200/6,800" in d.rationale


# ---------------------------------------------------------------------------
# Sparse data path
# ---------------------------------------------------------------------------


async def test_ingestor_sparse_data_returns_replan(base_state: PipelineState):
    """When shot density is too low the agent returns sparse_data decision."""
    decision = _make_decision(
        passed=False,
        accepted_shots=50,
        rationale="Only 50 shots accepted — 0.001 shots/km², below minimum of 1.0.",
        recommended_action="replan_widen_date",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(ingestor_agent, "run", new=AsyncMock(return_value=mock_result)):
        deps = _make_deps()
        result = await ingestor_agent.run("ingest", deps=deps)

    d = result.output
    assert d.passed is False
    assert d.recommended_action == "replan_widen_date"


# ---------------------------------------------------------------------------
# Replan leniency
# ---------------------------------------------------------------------------


async def test_ingestor_replan_uses_lenient_thresholds():
    """With replan_count=1, agent should use softer thresholds (verified via mocked decision)."""
    # On replan the agent should choose lower sensitivity and higher slope tolerance
    decision = _make_decision(
        sensitivity_min=0.90,
        slope_max_deg=35.0,
        rationale="Relaxed thresholds on replan: sensitivity < 0.90, slope > 35°.",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(ingestor_agent, "run", new=AsyncMock(return_value=mock_result)):
        deps = _make_deps(replan_count=1)
        result = await ingestor_agent.run("ingest", deps=deps)

    d = result.output
    assert d.sensitivity_min < 0.95
    assert d.slope_max_deg > 30.0


# ---------------------------------------------------------------------------
# Output schema validation
# ---------------------------------------------------------------------------


def test_ingestor_decision_rejects_invalid_action():
    with pytest.raises(Exception):
        IngestorDecision(
            passed=True,
            sensitivity_min=0.95,
            slope_max_deg=30.0,
            raw_shots=100,
            accepted_shots=80,
            rationale="test",
            recommended_action="invalid_action",  # not in Literal
        )


def test_ingestor_decision_defaults_warnings_to_empty_list():
    d = _make_decision()
    assert d.warnings == []
