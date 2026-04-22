import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from datetime import datetime

import pytest
from sqlalchemy.orm import sessionmaker

from canopy_height_prediction.db import AgentDecision, GediShotRaw, PipelineRun


def test_pipeline_run_created(test_db):
    _, TestSession = test_db
    with TestSession() as session:
        run = PipelineRun(
            run_id="int-test-001",
            aoi_bbox={"bbox": [-122.5, 37.5, -121.5, 38.5]},
            date_range={"start": "2025-01-01", "end": "2025-03-01"},
            status="running",
        )
        session.add(run)
        session.commit()

        fetched = session.get(PipelineRun, "int-test-001")
        assert fetched is not None
        assert fetched.status == "running"


def test_gedi_shots_raw_written(test_db):
    _, TestSession = test_db
    with TestSession() as session:
        session.add(PipelineRun(
            run_id="int-test-002",
            aoi_bbox={},
            date_range={},
            status="running",
        ))
        shots = [
            GediShotRaw(
                run_id="int-test-002",
                shot_id=f"shot_{i}",
                lat=37.5 + i * 0.01,
                lon=-122.0 + i * 0.01,
                rh98=20.0 + i,
                sensitivity=0.97,
                slope=10.0,
                beam="BEAM0000",
                quality_flag=1,
            )
            for i in range(10)
        ]
        session.add_all(shots)
        session.commit()

        count = session.query(GediShotRaw).filter_by(run_id="int-test-002").count()
        assert count == 10


def test_agent_decision_written(test_db):
    _, TestSession = test_db
    with TestSession() as session:
        session.add(PipelineRun(
            run_id="int-test-003",
            aoi_bbox={},
            date_range={},
            status="running",
        ))
        decision = AgentDecision(
            run_id="int-test-003",
            agent="ingestor",
            replan_count=0,
            passed=True,
            rationale="Accepted 4,200/6,800 shots — rejected 38% sensitivity < 0.95.",
            recommended_action="proceed",
        )
        session.add(decision)
        session.commit()

        fetched = session.query(AgentDecision).filter_by(run_id="int-test-003").first()
        assert fetched.passed is True
        assert fetched.recommended_action == "proceed"
