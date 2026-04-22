from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineState:
    # Config
    run_id: str
    aoi_bbox: tuple          # (min_lon, min_lat, max_lon, max_lat)
    date_range: tuple        # ("2022-01-01", "2023-12-31")
    aoi_area_km2: float = 0.0

    # Ingestor
    raw_shots: int = 0
    accepted_shots: int = 0
    ingestor_rationale: str = ""
    ingestor_passed: bool = False
    ingestor_recommended_action: str = ""

    # Transformer
    cv_block_size_km: Optional[float] = None
    n_folds: int = 5
    transformer_rationale: str = ""
    transformer_passed: bool = False
    transformer_recommended_action: str = ""

    # QA
    qa_rationale: str = ""
    qa_passed: bool = False
    qa_issues: list = field(default_factory=list)
    qa_recommended_action: str = ""

    # Post-training (written back manually after XGBoost)
    r2: Optional[float] = None
    rmse: Optional[float] = None

    # Orchestrator
    replan_count: int = 0
    decision_log: list = field(default_factory=list)
