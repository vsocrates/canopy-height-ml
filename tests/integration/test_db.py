import os

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from datetime import datetime

import pytest
from sqlalchemy.orm import sessionmaker

from canopy_height_prediction.db import AgentDecision, GediShotCleaned, GediShotRaw, PipelineRun


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


def test_gedi_shot_cleaned_written_with_scl_valid_obs(test_db):
    _, TestSession = test_db
    with TestSession() as session:
        session.add(PipelineRun(
            run_id="int-test-004",
            aoi_bbox={},
            date_range={},
            status="running",
        ))
        shots = [
            GediShotCleaned(
                run_id="int-test-004",
                shot_id=f"shot_{i}",
                lat=37.5 + i * 0.01,
                lon=-122.0 + i * 0.01,
                rh98=20.0 + i,
                b8=2500.0,
                b4=800.0,
                ndvi=0.51,
                evi=0.38,
                scl_valid_obs=5 + i,
                fold=i % 5,
                split="train" if i % 5 != 0 else "test",
            )
            for i in range(10)
        ]
        session.add_all(shots)
        session.commit()

        rows = session.query(GediShotCleaned).filter_by(run_id="int-test-004").all()
        assert len(rows) == 10
        assert all(r.scl_valid_obs is not None for r in rows)
        assert rows[0].scl_valid_obs == 5


def test_gedi_shot_cleaned_scl_valid_obs_nullable(test_db):
    _, TestSession = test_db
    with TestSession() as session:
        session.add(PipelineRun(
            run_id="int-test-005",
            aoi_bbox={},
            date_range={},
            status="running",
        ))
        shot = GediShotCleaned(
            run_id="int-test-005",
            shot_id="shot_null_obs",
            lat=37.5,
            lon=-122.0,
            rh98=15.0,
            scl_valid_obs=None,  # explicitly null — shot over persistent cloud
            fold=0,
            split="test",
        )
        session.add(shot)
        session.commit()

        fetched = session.query(GediShotCleaned).filter_by(run_id="int-test-005").first()
        assert fetched.scl_valid_obs is None
