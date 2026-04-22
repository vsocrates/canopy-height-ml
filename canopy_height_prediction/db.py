from contextlib import contextmanager
from datetime import datetime
import os

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///pipeline.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    aoi_bbox: Mapped[dict] = mapped_column(JSON, nullable=False)
    date_range: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String, default="running")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GediShotRaw(Base):
    __tablename__ = "gedi_shots_raw"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("pipeline_runs.run_id"), nullable=False)
    shot_id: Mapped[str] = mapped_column(String, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    rh98: Mapped[float] = mapped_column(Float, nullable=True)
    sensitivity: Mapped[float] = mapped_column(Float, nullable=True)
    slope: Mapped[float] = mapped_column(Float, nullable=True)
    beam: Mapped[str] = mapped_column(String, nullable=True)
    quality_flag: Mapped[int] = mapped_column(Integer, nullable=True)


class GediShotCleaned(Base):
    __tablename__ = "gedi_shots_cleaned"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("pipeline_runs.run_id"), nullable=False)
    shot_id: Mapped[str] = mapped_column(String, nullable=False)
    lat: Mapped[float] = mapped_column(Float, nullable=False)
    lon: Mapped[float] = mapped_column(Float, nullable=False)
    rh98: Mapped[float] = mapped_column(Float, nullable=True)
    b2: Mapped[float] = mapped_column(Float, nullable=True)
    b3: Mapped[float] = mapped_column(Float, nullable=True)
    b4: Mapped[float] = mapped_column(Float, nullable=True)
    b8: Mapped[float] = mapped_column(Float, nullable=True)
    b11: Mapped[float] = mapped_column(Float, nullable=True)
    b12: Mapped[float] = mapped_column(Float, nullable=True)
    ndvi: Mapped[float] = mapped_column(Float, nullable=True)
    evi: Mapped[float] = mapped_column(Float, nullable=True)
    fold: Mapped[int] = mapped_column(Integer, nullable=True)
    split: Mapped[str] = mapped_column(String, nullable=True)


class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("pipeline_runs.run_id"), nullable=False)
    agent: Mapped[str] = mapped_column(String, nullable=False)
    replan_count: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rationale: Mapped[str] = mapped_column(Text, nullable=False)
    recommended_action: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    Base.metadata.create_all(engine)
