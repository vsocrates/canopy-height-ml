import os

from dotenv import load_dotenv

load_dotenv()
# Fallback so pydantic-ai can instantiate the Anthropic provider at import
# time even without a .env. Tests mock agent.run() so this is never called.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from canopy_height_prediction.db import Base
from canopy_height_prediction.state import PipelineState


@pytest.fixture
def test_db(tmp_path):
    """In-memory SQLite DB per test via tmp_path."""
    db_path = tmp_path / "test_pipeline.db"
    test_engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(test_engine)
    TestSession = sessionmaker(bind=test_engine, expire_on_commit=False)
    yield test_engine, TestSession
    test_engine.dispose()


@pytest.fixture
def base_state():
    return PipelineState(
        run_id="test-run-001",
        aoi_bbox=(-122.5, 37.5, -121.5, 38.5),
        date_range=("2025-01-01", "2025-03-01"),
    )
