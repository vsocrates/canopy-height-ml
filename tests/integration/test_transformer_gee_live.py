"""
Integration tests for extract_sentinel_bands against real GEE + a real SQLite DB.

These hit the actual Earth Engine API — run only when explicitly requested:
    uv run pytest -m integration -v

Requires:
    - GOOGLE_CLOUD_PROJECT env var (or .env)
    - GEE credentials at ~/.config/earthengine/credentials
      (run `uv run python -c "import ee; ee.Authenticate()"` once if missing)

AOI: same ~100 km² Tahoe slice used by ingestor live tests.
Seeds a handful of synthetic raw shots so the tool has something to sample.
"""

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

import ee
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from canopy_height_prediction.agents import (
    MIN_VALID_OBS,
    TransformerDeps,
    extract_sentinel_bands,
)
from canopy_height_prediction.db import Base, GediShotCleaned, GediShotRaw, PipelineRun

SMALL_AOI = (-120.3, 39.3, -120.2, 39.4)
DATE_START = "2022-01-01"
DATE_END = "2023-01-01"

# A handful of real coordinates inside the AOI
SEED_SHOTS = [
    ("shot_001", 39.32, -120.28, 18.5),
    ("shot_002", 39.34, -120.26, 22.1),
    ("shot_003", 39.36, -120.24, 30.7),
    ("shot_004", 39.38, -120.22, 25.0),
    ("shot_005", 39.31, -120.27, 12.3),
]


class MockCtx:
    def __init__(self, deps):
        self.deps = deps


@pytest.fixture(scope="module", autouse=True)
def init_gee():
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "canopy-height-ml")
    ee.Initialize(project=project)


@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory):
    """SQLite DB seeded with a PipelineRun and SEED_SHOTS raw shots."""
    db_path = tmp_path_factory.mktemp("transformer_live") / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    run_id = "live-transformer-test"
    with Session() as session:
        session.add(PipelineRun(
            run_id=run_id,
            aoi_bbox={"bbox": list(SMALL_AOI)},
            date_range={"start": DATE_START, "end": DATE_END},
            status="running",
        ))
        for shot_id, lat, lon, rh98 in SEED_SHOTS:
            session.add(GediShotRaw(
                run_id=run_id,
                shot_id=shot_id,
                lat=lat,
                lon=lon,
                rh98=rh98,
                sensitivity=0.97,
                slope=10.0,
                beam="BEAM0000",
                quality_flag=1,
            ))
        session.commit()

    yield engine, Session, run_id
    engine.dispose()


def make_deps(run_id: str) -> TransformerDeps:
    return TransformerDeps(
        run_id=run_id,
        aoi_bbox=SMALL_AOI,
        date_start=DATE_START,
        date_end=DATE_END,
    )


@pytest.mark.integration
def test_extract_sentinel_bands_returns_expected_keys(seeded_db, monkeypatch):
    engine, Session, run_id = seeded_db

    # Point get_session at the test DB
    import canopy_height_prediction.db as db_module
    monkeypatch.setattr(db_module, "SessionLocal", Session)

    deps = make_deps(run_id)
    result = extract_sentinel_bands(MockCtx(deps))

    assert "error" not in result
    assert result["n_shots"] == len(SEED_SHOTS)
    assert "n_matched" in result
    assert "nan_rate_pct" in result
    assert "valid_obs_below_threshold_pct" in result
    assert "min_valid_obs_used" in result
    assert result["min_valid_obs_used"] == MIN_VALID_OBS


@pytest.mark.integration
def test_extract_sentinel_bands_populates_matched_df(seeded_db, monkeypatch):
    engine, Session, run_id = seeded_db

    import canopy_height_prediction.db as db_module
    monkeypatch.setattr(db_module, "SessionLocal", Session)

    deps = make_deps(run_id)
    extract_sentinel_bands(MockCtx(deps))

    df = deps._matched_df
    assert df is not None
    assert len(df) > 0
    assert "shot_id" in df.columns
    assert "rh98" in df.columns
    assert "valid_obs_count" in df.columns
    for band in ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "evi"]:
        assert band in df.columns, f"Expected band {band} in _matched_df"


@pytest.mark.integration
def test_extract_sentinel_bands_valid_obs_count_is_non_negative(seeded_db, monkeypatch):
    engine, Session, run_id = seeded_db

    import canopy_height_prediction.db as db_module
    monkeypatch.setattr(db_module, "SessionLocal", Session)

    deps = make_deps(run_id)
    extract_sentinel_bands(MockCtx(deps))

    df = deps._matched_df
    valid_counts = df["valid_obs_count"].dropna()
    assert (valid_counts >= 0).all(), "valid_obs_count must be non-negative"


@pytest.mark.integration
def test_extract_sentinel_bands_nan_rate_is_plausible(seeded_db, monkeypatch):
    """Over Tahoe with a full year of data, nan_rate should be well under 50%."""
    engine, Session, run_id = seeded_db

    import canopy_height_prediction.db as db_module
    monkeypatch.setattr(db_module, "SessionLocal", Session)

    deps = make_deps(run_id)
    result = extract_sentinel_bands(MockCtx(deps))

    assert result["nan_rate_pct"] < 50.0, (
        f"Expected low nan_rate over Tahoe, got {result['nan_rate_pct']}%"
    )
