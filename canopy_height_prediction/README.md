# Canopy Height Prediction — Agentic Pipeline

This package implements an agentic data pipeline that produces a spatially cross-validated, XGBoost-ready dataset pairing GEDI L2A canopy height shots with Sentinel-2 spectral features.

## Overview

Four LLM agents run in sequence, each with a narrow responsibility. The Orchestrator owns the loop and handles replanning when a stage fails.

```
Orchestrator
├── Ingestor   — query GEDI + S2, choose filter thresholds, write raw shots to DB
├── Transformer — match shots to S2 bands, estimate variogram, assign spatial CV folds
└── QA          — validate feature distributions, fold balance, and rh98 target range
```

If any stage fails (insufficient data, too many NaNs, imbalanced folds), the Orchestrator either widens the date window or relaxes quality thresholds and reruns from the failed stage. Up to `max_replans=3` attempts are made before aborting.

## Agents

### Ingestor (`agents/ingestor.py`)

Queries Earth Engine for GEDI L2A shots and Sentinel-2 scene availability over the AOI and date range. Reasons about filter thresholds from histogram data rather than using hard-coded defaults, then writes accepted shots to `gedi_shots_raw`.

**Tools:** `query_gedi_earthengine`, `query_sentinel2`, `compute_shot_density`, `apply_quality_filters`, `write_raw_shots_to_db`

**Fails with `replan_widen_date` when** filtered shot density falls below `min_shot_density_per_km2` (default 1.0 shots/km²).

**Default thresholds** (agent may adjust based on histograms):
- `sensitivity_min = 0.95`
- `slope_max_deg = 30.0`

### Transformer (`agents/transformer.py`)

Loads raw shots from the DB, samples a Sentinel-2 median composite (B2, B3, B4, B8, B11, B12 + NDVI, EVI) at each shot location via batched GEE `sampleRegions` calls, then estimates the spatial autocorrelation range via an experimental variogram. Chooses a CV block size that exceeds the variogram range (≥1.5×) to prevent spatial leakage, assigns shots to spatial folds, and writes to `gedi_shots_cleaned`.

**Tools:** `extract_sentinel_bands`, `compute_variogram`, `generate_spatial_blocks`, `assign_folds`, `write_cleaned_shots_to_db`

**Fails with `replan_widen_date` when** NaN rate > 20% or > 50% of shots have fewer than 2 clear-sky scenes.

**GEE extraction constants:**
- `_MAX_EXTRACTION_SHOTS = 5000` — shots beyond this are randomly subsampled
- `_SAMPLE_BATCH = 500` — shots per `sampleRegions` request (keeps payload < 10 MB)
- `_SAMPLE_WORKERS = 8` — concurrent GEE batch requests

### QA (`agents/qa.py`)

Reads the cleaned dataset from `gedi_shots_cleaned` and checks it for issues that would invalidate model training.

**Tools:** `read_cleaned_shots_from_db`, `check_feature_distributions`, `check_fold_balance`, `check_target_range`

**Flags:**
- Any band with NaN rate > 10%
- Any fold with < 5% or > 40% of total shots
- rh98 ≤ 0 in > 5% of shots (physically impossible)
- rh98 > 60 m in > 10% of shots (implausible for most forest types)

### Orchestrator (`agents/orchestrator.py`)

Drives the full pipeline loop. Calls each sub-agent in order, evaluates pass/fail, and either proceeds or replans.

**Tools:** `run_ingestor`, `run_transformer`, `run_qa`, `replan`, `abort`

**Replan strategies:**
- `replan_widen_date` → extends `date_end` by 90 days
- `replan_relax_thresholds` → reduces `sensitivity_min` by 0.05, increases `slope_max_deg` by 5°

## Package Layout

```
canopy_height_prediction/
  agents/
    __init__.py          # empty
    ingestor.py
    transformer.py
    qa.py
    orchestrator.py
  _helpers.py            # GEE utilities: _load_gedi_shots, _fc_to_gdf, _estimate_variogram_range
  _models.py             # Pydantic Deps + Decision models for all agents
  db.py                  # SQLAlchemy models: PipelineRun, GediShotRaw, GediShotCleaned
  state.py               # PipelineState dataclass (accumulates results across stages)
  runners.py             # Thin async wrappers that map PipelineState ↔ agent Deps/Decision
  plots.py               # Diagnostic figures: shot map, fold map, variogram, distributions
  main.py                # CLI entrypoint
```

## Running

```bash
uv run python -m canopy_height_prediction.main \
    --bbox=-122.5,37.5,-121.5,38.5 \
    --date-start=2023-01-01 \
    --date-end=2023-06-01 \
    [--run-id my-run] \
    [--n-folds 5] \
    [--output results.json] \
    [--save-figs ./figures]
```

Requires:
- `ANTHROPIC_API_KEY` in environment or `.env`
- `GOOGLE_CLOUD_PROJECT` with Earth Engine access
- GEE credentials at `~/.config/earthengine/credentials`

## Database

SQLite by default (`pipeline.db`). Override with `DATABASE_URL` env var.

| Table | Contents |
|---|---|
| `pipeline_run` | One row per run: AOI, date range, status |
| `gedi_shots_raw` | Filtered GEDI shots written by Ingestor |
| `gedi_shots_cleaned` | Band-matched, folded shots written by Transformer |
| `agent_decision` | Full decision log for every agent invocation |

## Data Flow

```
GEE (GEDI L2A)  ──┐
                   ├──► Ingestor ──► gedi_shots_raw
GEE (Sentinel-2) ─┘
                        │
                        ▼
                   Transformer ──► gedi_shots_cleaned
                   (S2 bands, spatial folds)
                        │
                        ▼
                        QA ──► pass/fail + issues
                        │
                        ▼
                   XGBoost training (downstream)
```
