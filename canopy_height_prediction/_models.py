from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# GEE Tool Return Models
# ---------------------------------------------------------------------------


class GediQueryResult(BaseModel):
    """Return type of query_gedi_earthengine tool."""

    total_shots: int
    aoi_area_km2: float
    raw_shot_density_per_km2: float
    # Each bin is [bin_center, count] — 10 bins per histogram
    rh98_histogram: list[list]
    sensitivity_histogram: list[list]
    slope_histogram: list[list]
    quality_flag_counts: dict[str, int]


class Sentinel2QueryResult(BaseModel):
    """Return type of query_sentinel2 tool."""

    scene_count: int
    mean_cloud_cover_pct: float
    temporal_coverage_days: Optional[int]
    bands_available: list[str]


# ---------------------------------------------------------------------------
# Ingestor — Deps + Output
# ---------------------------------------------------------------------------


@dataclass
class IngestorDeps:
    run_id: str
    aoi_bbox: tuple                          # (min_lon, min_lat, max_lon, max_lat)
    date_start: str
    date_end: str
    replan_count: int = 0                    # agent is more lenient when > 0
    sensitivity_min_suggested: float = 0.95
    slope_max_deg_suggested: float = 30.0
    min_shot_density_per_km2: float = 1.0
    gee_project: str = "canopy-height-ml"
    # Populated by apply_quality_filters tool so write_raw_shots_to_db can access it
    _filtered_shots: Optional[object] = field(default=None, repr=False)


class IngestorDecision(BaseModel):
    passed: bool
    sensitivity_min: float
    slope_max_deg: float
    raw_shots: int
    accepted_shots: int
    rationale: str
    recommended_action: Literal[
        "proceed",
        "replan_widen_date",
        "replan_relax_thresholds",
        "abort",
    ]
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Transformer — Deps + Output
# ---------------------------------------------------------------------------


@dataclass
class TransformerDeps:
    run_id: str
    aoi_bbox: tuple
    date_start: str
    date_end: str
    replan_count: int = 0
    n_folds: int = 5
    gee_project: str = "canopy-height-ml"
    # Populated by extract_sentinel_bands; consumed by compute_variogram / assign_folds
    _matched_df: Optional[object] = field(default=None, repr=False)
    # Populated by assign_folds; consumed by write_cleaned_shots_to_db
    _folded_df: Optional[object] = field(default=None, repr=False)


class TransformerDecision(BaseModel):
    passed: bool
    cv_block_size_km: float
    n_folds: int
    rationale: str
    recommended_action: Literal[
        "proceed",
        "replan_widen_date",
        "replan_relax_thresholds",
        "abort",
    ]
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# QA — Deps + Output
# ---------------------------------------------------------------------------


@dataclass
class QADeps:
    run_id: str
    replan_count: int = 0
    r2_threshold: float = 0.6
    rmse_threshold_m: float = 5.0
    # Populated by read_cleaned_shots_from_db; consumed by check_* tools
    _cleaned_df: Optional[object] = field(default=None, repr=False)


class QADecision(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    rationale: str
    recommended_action: Literal[
        "proceed",
        "replan_widen_date",
        "replan_relax_thresholds",
        "abort",
    ]


# ---------------------------------------------------------------------------
# Orchestrator — Deps + Output
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorDeps:
    run_id: str
    aoi_bbox: tuple
    date_start: str
    date_end: str
    max_replans: int = 3
    replan_count: int = 0
    # Mutable thresholds — updated in place by the replan tool
    sensitivity_min: float = 0.95
    slope_max_deg: float = 30.0
    n_folds: int = 5
    min_shot_density_per_km2: float = 1.0
    gee_project: str = "canopy-height-ml"


class OrchestratorDecision(BaseModel):
    action: Literal["proceed", "replan", "abort"]
    rationale: str
    final_date_start: Optional[str] = None
    final_date_end: Optional[str] = None
    final_sensitivity_min: Optional[float] = None
    final_slope_max_deg: Optional[float] = None
    replan_count: int = 0
