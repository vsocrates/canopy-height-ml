"""
Integration tests for the Orchestrator tools.

Sub-agents (ingestor, transformer, qa) are mocked at the agent.run level so no
GEE or LLM calls are made. A real SQLite DB via tmp_path verifies that the
orchestrator tools correctly wire up deps, create DB records, and mutate state.

Run with: uv run pytest tests/integration/test_orchestrator_tools.py -v
"""

import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from canopy_height_prediction._models import (
    IngestorDecision,
    OrchestratorDeps,
    QADecision,
    TransformerDecision,
)
from canopy_height_prediction.agents.ingestor import ingestor_agent
from canopy_height_prediction.agents.orchestrator import abort, replan, run_ingestor, run_qa, run_transformer
from canopy_height_prediction.agents.qa import qa_agent
from canopy_height_prediction.agents.transformer import transformer_agent
from canopy_height_prediction.db import Base, PipelineRun


class MockCtx:
    def __init__(self, deps):
        self.deps = deps


def make_deps(tmp_path, **overrides) -> OrchestratorDeps:
    defaults = dict(
        run_id="orch-int-test",
        aoi_bbox=(-122.5, 37.5, -121.5, 38.5),
        date_start="2023-01-01",
        date_end="2023-04-01",
        max_replans=3,
        replan_count=0,
    )
    defaults.update(overrides)
    return OrchestratorDeps(**defaults)


def _ingestor_decision(passed=True, action="proceed", accepted=3200) -> IngestorDecision:
    return IngestorDecision(
        passed=passed,
        sensitivity_min=0.95,
        slope_max_deg=30.0,
        raw_shots=5000,
        accepted_shots=accepted,
        rationale=f"Accepted {accepted}/5000 shots.",
        recommended_action=action,
    )


def _transformer_decision(passed=True, action="proceed") -> TransformerDecision:
    return TransformerDecision(
        passed=passed,
        cv_block_size_km=12.0,
        n_folds=5,
        rationale="Variogram range ~8km; block=12km. 5 balanced folds.",
        recommended_action=action,
    )


def _qa_decision(passed=True, action="proceed") -> QADecision:
    return QADecision(
        passed=passed,
        issues=[],
        rationale="rh98 [1.2, 55m], folds balanced, no NaN issues.",
        recommended_action=action,
    )


@pytest.fixture()
def patched_db(tmp_path, monkeypatch):
    """Real SQLite DB wired into get_session for the duration of the test."""
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    import canopy_height_prediction.db as db_module
    monkeypatch.setattr(db_module, "SessionLocal", Session)

    yield engine, Session


# ---------------------------------------------------------------------------
# run_ingestor creates the PipelineRun record
# ---------------------------------------------------------------------------


async def test_run_ingestor_creates_pipeline_run_record(patched_db):
    engine, Session = patched_db
    deps = make_deps(tmp_path=None)

    mock_result = MagicMock()
    mock_result.output = _ingestor_decision()

    with patch.object(ingestor_agent, "run", new=AsyncMock(return_value=mock_result)):
        result = await run_ingestor(MockCtx(deps))

    assert result["passed"] is True
    with Session() as session:
        run = session.get(PipelineRun, "orch-int-test")
    assert run is not None
    assert run.status == "running"


# ---------------------------------------------------------------------------
# Happy path — all three agents pass
# ---------------------------------------------------------------------------


async def test_happy_path_all_agents_pass(patched_db):
    engine, Session = patched_db
    deps = make_deps(tmp_path=None)

    with patch.object(ingestor_agent, "run", new=AsyncMock(
        return_value=MagicMock(output=_ingestor_decision(passed=True))
    )), patch.object(transformer_agent, "run", new=AsyncMock(
        return_value=MagicMock(output=_transformer_decision(passed=True))
    )), patch.object(qa_agent, "run", new=AsyncMock(
        return_value=MagicMock(output=_qa_decision(passed=True))
    )):
        ingestor_result = await run_ingestor(MockCtx(deps))
        transformer_result = await run_transformer(MockCtx(deps))
        qa_result = await run_qa(MockCtx(deps))

    assert ingestor_result["passed"] is True
    assert transformer_result["passed"] is True
    assert qa_result["passed"] is True
    assert deps.replan_count == 0


# ---------------------------------------------------------------------------
# Ingestor fails → replan → ingestor passes on retry
# ---------------------------------------------------------------------------


async def test_ingestor_fails_replan_then_passes(patched_db):
    engine, Session = patched_db
    deps = make_deps(tmp_path=None, date_end="2023-04-01")

    # First call fails, second passes
    fail_result = MagicMock(output=_ingestor_decision(passed=False, action="replan_widen_date", accepted=5))
    pass_result = MagicMock(output=_ingestor_decision(passed=True, accepted=3200))

    with patch.object(ingestor_agent, "run", new=AsyncMock(side_effect=[fail_result, pass_result])):
        first = await run_ingestor(MockCtx(deps))
        assert first["passed"] is False

        replan_result = replan(MockCtx(deps), reason="too few shots", extend_days=90)
        assert replan_result["replan_count"] == 1
        assert deps.date_end == "2023-06-30"

        second = await run_ingestor(MockCtx(deps))
        assert second["passed"] is True

    assert deps.replan_count == 1


# ---------------------------------------------------------------------------
# Max replans exceeded → abort
# ---------------------------------------------------------------------------


async def test_max_replans_exceeded_triggers_abort(patched_db):
    engine, Session = patched_db
    deps = make_deps(tmp_path=None, max_replans=2, replan_count=2)

    # replan should refuse; abort should fire
    replan_result = replan(MockCtx(deps), reason="still failing", extend_days=90)
    assert "error" in replan_result

    abort_result = abort(MockCtx(deps), reason="max_replans=2 reached — giving up")
    assert abort_result["aborted"] is True
    assert abort_result["replan_count"] == 2
