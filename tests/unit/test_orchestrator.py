import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from canopy_height_prediction._models import OrchestratorDecision, OrchestratorDeps
from canopy_height_prediction.agents.orchestrator import abort, orchestrator_agent, replan


class MockCtx:
    def __init__(self, deps):
        self.deps = deps


def make_deps(**overrides) -> OrchestratorDeps:
    defaults = dict(
        run_id="test-run-001",
        aoi_bbox=(-122.5, 37.5, -121.5, 38.5),
        date_start="2023-01-01",
        date_end="2023-04-01",
        max_replans=3,
        replan_count=0,
    )
    defaults.update(overrides)
    return OrchestratorDeps(**defaults)


def _make_decision(**overrides) -> OrchestratorDecision:
    defaults = dict(
        action="proceed",
        rationale=(
            "Ingestor: 3,200 shots accepted. "
            "Transformer: nan_rate=4%, block=12km. "
            "QA: rh98 [1.2, 55m], folds balanced — no issues."
        ),
        final_date_start="2023-01-01",
        final_date_end="2023-04-01",
        final_sensitivity_min=0.95,
        final_slope_max_deg=30.0,
        replan_count=0,
    )
    defaults.update(overrides)
    return OrchestratorDecision(**defaults)


# ---------------------------------------------------------------------------
# replan tool — pure function
# ---------------------------------------------------------------------------


def test_replan_increments_replan_count():
    deps = make_deps(replan_count=0)
    result = replan(MockCtx(deps), reason="too few shots", extend_days=0)
    assert result["replan_count"] == 1
    assert deps.replan_count == 1


def test_replan_extends_date_end():
    deps = make_deps(date_end="2023-04-01")
    replan(MockCtx(deps), reason="widen date", extend_days=90)
    assert deps.date_end == "2023-06-30"


def test_replan_relaxes_thresholds():
    deps = make_deps(sensitivity_min=0.95, slope_max_deg=30.0)
    replan(MockCtx(deps), reason="relax filters", relax_thresholds=True)
    assert deps.sensitivity_min == pytest.approx(0.90, abs=0.01)
    assert deps.slope_max_deg == pytest.approx(35.0, abs=0.1)


def test_replan_clamps_thresholds_at_limits():
    deps = make_deps(sensitivity_min=0.72, slope_max_deg=43.0)
    replan(MockCtx(deps), reason="relax again", relax_thresholds=True)
    assert deps.sensitivity_min >= 0.70
    assert deps.slope_max_deg <= 45.0


def test_replan_returns_error_when_max_replans_reached():
    deps = make_deps(replan_count=3, max_replans=3)
    result = replan(MockCtx(deps), reason="try again", extend_days=90)
    assert "error" in result
    assert deps.replan_count == 3  # not incremented


# ---------------------------------------------------------------------------
# abort tool — pure function
# ---------------------------------------------------------------------------


def test_abort_returns_correct_fields():
    deps = make_deps(replan_count=2, date_end="2023-07-01")
    result = abort(MockCtx(deps), reason="data fundamentally unusable")
    assert result["aborted"] is True
    assert result["reason"] == "data fundamentally unusable"
    assert result["replan_count"] == 2
    assert result["date_end"] == "2023-07-01"


# ---------------------------------------------------------------------------
# OrchestratorDecision schema
# ---------------------------------------------------------------------------


def test_orchestrator_decision_rejects_invalid_action():
    with pytest.raises(Exception):
        OrchestratorDecision(action="maybe", rationale="test")


# ---------------------------------------------------------------------------
# Agent-level decision shapes
# ---------------------------------------------------------------------------


async def test_orchestrator_happy_path_returns_proceed():
    decision = _make_decision()
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(orchestrator_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await orchestrator_agent.run("run pipeline", deps=make_deps())

    d = result.output
    assert d.action == "proceed"
    assert d.replan_count == 0
    assert "Ingestor" in d.rationale


async def test_orchestrator_ingestor_failure_leads_to_replan():
    decision = _make_decision(
        action="replan",
        rationale="Ingestor failed: 0.4 shots/km² below threshold. Replanned with +90 days.",
        replan_count=1,
        final_date_end="2023-07-01",
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(orchestrator_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await orchestrator_agent.run("run pipeline", deps=make_deps())

    d = result.output
    assert d.action == "replan"
    assert d.replan_count == 1


async def test_orchestrator_max_replans_exceeded_aborts():
    decision = _make_decision(
        action="abort",
        rationale="Ingestor failed 3 times — max_replans reached. Aborting.",
        replan_count=3,
    )
    mock_result = MagicMock()
    mock_result.output = decision

    with patch.object(orchestrator_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await orchestrator_agent.run("run pipeline", deps=make_deps(max_replans=3))

    assert result.output.action == "abort"
    assert result.output.replan_count == 3
