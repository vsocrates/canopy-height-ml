"""
Integration tests for runners.py.

Verifies that each runner correctly maps PipelineState → agent Deps
and writes agent Decision fields back to PipelineState. Agent.run()
is mocked — no GEE or LLM calls are made.
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from canopy_height_prediction.agents import (
    ingestor_agent,
    orchestrator_agent,
    qa_agent,
    transformer_agent,
)
from canopy_height_prediction.runners import run_ingestor, run_orchestrator, run_qa, run_transformer
from canopy_height_prediction.state import PipelineState


def _make_state(**overrides) -> PipelineState:
    defaults = dict(
        run_id="runner-test",
        aoi_bbox=(-122.5, 37.5, -121.5, 38.5),
        date_range=("2023-01-01", "2023-06-01"),
        replan_count=0,
        n_folds=5,
    )
    defaults.update(overrides)
    return PipelineState(**defaults)


# ---------------------------------------------------------------------------
# run_ingestor
# ---------------------------------------------------------------------------


async def test_run_ingestor_writes_state_fields():
    state = _make_state()
    mock_output = MagicMock(
        passed=True,
        raw_shots=5000,
        accepted_shots=3200,
        rationale="Accepted 3200/5000 shots.",
        recommended_action="proceed",
        warnings=[],
        sensitivity_min=0.95,
        slope_max_deg=30.0,
    )
    mock_output.model_dump.return_value = {
        "passed": True, "raw_shots": 5000, "accepted_shots": 3200,
        "rationale": "Accepted 3200/5000 shots.", "recommended_action": "proceed",
        "warnings": [], "sensitivity_min": 0.95, "slope_max_deg": 30.0,
    }

    with patch.object(ingestor_agent, "run", new=AsyncMock(return_value=MagicMock(output=mock_output))):
        result = await run_ingestor(state)

    assert result.ingestor_passed is True
    assert result.raw_shots == 5000
    assert result.accepted_shots == 3200
    assert result.ingestor_rationale == "Accepted 3200/5000 shots."
    assert result.ingestor_recommended_action == "proceed"
    assert result.decision_log[-1]["agent"] == "ingestor"


# ---------------------------------------------------------------------------
# run_transformer
# ---------------------------------------------------------------------------


async def test_run_transformer_writes_state_fields():
    state = _make_state()
    mock_output = MagicMock(
        passed=True,
        cv_block_size_km=12.0,
        n_folds=5,
        rationale="Variogram range ~8km; block=12km.",
        recommended_action="proceed",
        warnings=[],
    )
    mock_output.model_dump.return_value = {
        "passed": True, "cv_block_size_km": 12.0, "n_folds": 5,
        "rationale": "Variogram range ~8km; block=12km.",
        "recommended_action": "proceed", "warnings": [],
    }

    with patch.object(transformer_agent, "run", new=AsyncMock(return_value=MagicMock(output=mock_output))):
        result = await run_transformer(state)

    assert result.transformer_passed is True
    assert result.cv_block_size_km == 12.0
    assert result.n_folds == 5
    assert result.decision_log[-1]["agent"] == "transformer"


# ---------------------------------------------------------------------------
# run_qa
# ---------------------------------------------------------------------------


async def test_run_qa_writes_state_fields():
    state = _make_state()
    mock_output = MagicMock(
        passed=True,
        issues=[],
        rationale="rh98 [1.2, 55m], folds balanced.",
        recommended_action="proceed",
    )
    mock_output.model_dump.return_value = {
        "passed": True, "issues": [],
        "rationale": "rh98 [1.2, 55m], folds balanced.",
        "recommended_action": "proceed",
    }

    with patch.object(qa_agent, "run", new=AsyncMock(return_value=MagicMock(output=mock_output))):
        result = await run_qa(state)

    assert result.qa_passed is True
    assert result.qa_issues == []
    assert result.qa_rationale == "rh98 [1.2, 55m], folds balanced."
    assert result.decision_log[-1]["agent"] == "qa"


# ---------------------------------------------------------------------------
# run_orchestrator — date_range write-back
# ---------------------------------------------------------------------------


async def test_run_orchestrator_writes_back_replanned_dates():
    state = _make_state(date_range=("2023-01-01", "2023-06-01"))
    mock_output = MagicMock(
        replan_count=1,
        final_date_start=None,
        final_date_end="2023-09-01",
    )
    mock_output.model_dump.return_value = {
        "action": "proceed", "rationale": "ok", "replan_count": 1,
        "final_date_start": None, "final_date_end": "2023-09-01",
        "final_sensitivity_min": None, "final_slope_max_deg": None,
    }

    with patch.object(orchestrator_agent, "run", new=AsyncMock(return_value=MagicMock(output=mock_output))):
        result = await run_orchestrator(state)

    assert result.replan_count == 1
    assert result.date_range == ("2023-01-01", "2023-09-01")
    assert result.decision_log[-1]["agent"] == "orchestrator"
