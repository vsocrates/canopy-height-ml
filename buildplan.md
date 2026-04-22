# Fuel Load Estimation — Agentic Pipeline Spec

## Project Goal
Agentic data pipeline that ingests GEDI L2A lidar + Sentinel-2 imagery, prepares spatially-aware training data, and hands off a validated dataset for XGBoost canopy height regression (rh98 as target variable). Training is done manually outside the pipeline.

---

## Tech Stack

| Layer | Tool | Purpose |
|---|---|---|
| Agent framework | PydanticAI | Model calls, tool registration + execution, typed structured outputs |
| LLM | claude-sonnet-4-20250514 | All agent reasoning |
| Observability | Logfire | Auto-instrumented PydanticAI traces + manual spans on non-agent steps |
| Database | SQLite via SQLAlchemy | Stores raw shots, cleaned data, agent decisions, run metadata |
| Migrations | Alembic | Versioned schema changes |
| State | Python dataclass (PipelineState) | Single object accumulated across full pipeline |
| Data I/O | Pandas | DataFrame manipulation between pipeline steps |
| Remote sensing | Earth Engine Python API | GEDI L2A + Sentinel-2 queries (called via agent tools) |
| Spatial CV | scikit-learn + custom variogram | Block-based spatial cross-validation (called via agent tools) |
| Testing | pytest + pytest-asyncio + pytest-mock | Unit + integration tests, async support |
| Dependency mgmt | pip + requirements.txt | Standard |

---

## Agents

### Agent 1 — Orchestrator
- Owns the main pipeline loop (max 3 replans)
- Receives pass/fail + recommended_action from each downstream agent
- Decides whether to proceed, replan with adjusted parameters, or abort
- On replan: mutates PipelineState (e.g. widens date_range, relaxes thresholds) and reruns from the failed step
- Tools: `run_ingestor`, `run_transformer`, `run_qa`, `replan`, `abort`

### Agent 2 — Ingestor
- Queries GEDI L2A and Sentinel-2 via Earth Engine tools
- Reasons about data quality: sensitivity thresholds, slope thresholds, shot density
- Writes rationale for every accept/reject decision
- Writes raw shots to SQLite via tool
- Returns pass/fail + recommended_action to orchestrator
- If replan_count > 0, is more lenient with thresholds
- Tools: `query_gedi_earthengine`, `query_sentinel2`, `compute_shot_density`, `apply_quality_filters`, `write_raw_shots_to_db`

### Agent 3 — Transformer
- Matches GEDI footprints to Sentinel-2 band values at same locations
- Computes variogram to estimate spatial autocorrelation range
- Reasons about appropriate spatial CV block size (must exceed variogram range)
- Outputs justified train/test fold assignments
- Writes cleaned + folded dataset to SQLite via tool
- Tools: `compute_variogram`, `generate_spatial_blocks`, `assign_folds`, `extract_sentinel_bands`, `write_cleaned_shots_to_db`

### Agent 4 — QA / Validator
- Reads cleaned dataset from SQLite via tool
- Inspects for: target leakage, spatial fold imbalance, physically implausible rh98 values (negative or >60m in non-forest), missing S2 bands
- Writes issues list + pass/fail rationale
- Returns recommended_action to orchestrator
- Tools: `read_cleaned_shots_from_db`, `check_feature_distributions`, `check_fold_balance`, `check_target_range`

---

## Shared State — PipelineState (dataclass)

```
run_id, aoi_bbox, date_range, aoi_area_km2
raw_shots, accepted_shots
ingestor_rationale, ingestor_passed, ingestor_recommended_action
cv_block_size_km, n_folds
transformer_rationale, transformer_passed, transformer_recommended_action
qa_rationale, qa_passed, qa_issues[], qa_recommended_action
r2, rmse (post-training, written back manually)
replan_count, decision_log[]
```

Single object passed through entire pipeline. Agents receive `deps` (ephemeral per-call context) not the state object directly. Runners read agent results and write back to PipelineState.

---

## Database Schema (SQLite via SQLAlchemy + Alembic)

- **pipeline_runs** — run_id, aoi_bbox, date_range, status, created_at
- **gedi_shots_raw** — run_id, shot_id, lat, lon, rh98, sensitivity, slope, beam, quality_flag
- **gedi_shots_cleaned** — run_id, shot_id, lat, lon, rh98, B2, B3, B4, B8, B11, B12, NDVI, EVI, fold, split
- **agent_decisions** — run_id, agent, replan_count, passed, rationale, recommended_action, created_at

---

## File Structure

```
project/
  agents.py          # All agent definitions, Deps dataclasses, output BaseModels, tool registrations
  state.py           # PipelineState dataclass
  runners.py         # Thin async runners — build deps, call agent, write result to state
  orchestrator.py    # Main pipeline loop + replanning logic
  db.py              # SQLAlchemy engine, session, Base, init_db
  migrations/        # Alembic migration versions
  tests/
    conftest.py      # Shared fixtures: fake DataFrames, test DB via tmp_path, base PipelineState
    unit/
      test_ingestor.py
      test_transformer.py
      test_qa.py
      test_orchestrator.py
    integration/
      test_db.py
      test_pipeline.py
  main.py            # Entrypoint + CLI arg parsing
```

Note: No separate ee_queries.py or spatial.py — those functions live as `@agent.tool` registrations inside agents.py.

---

## Commit Plan

### Commit 1 — `chore: project scaffold`
requirements.txt, .env.example, .gitignore, empty module files, pytest.ini with asyncio_mode=auto, Logfire config stub

### Commit 2 — `feat: database schema and migrations`
db.py (SQLAlchemy engine + session + Base), Alembic init, migration 001 with all four tables

### Commit 3 — `feat: pipeline state dataclass`
state.py only — PipelineState with all fields, no logic

### Commit 4 — `feat: agent output types and deps`
agents.py (partial) — all Pydantic BaseModel output types and all Deps dataclasses. No agent instantiation yet. Includes unit tests asserting field validation.

### Commit 5 — `feat: ingestor agent and tools`
agents.py (ingestor) — ingestor_agent instantiated with system prompt, all ingestor tools registered via @ingestor_agent.tool: EE query, quality filter, shot density, DB write. Unit tests with mocked EE + mocked DB.

### Commit 6 — `feat: transformer agent and tools`
agents.py (transformer) — transformer_agent with all tools: variogram, spatial blocks, fold assignment, S2 band extraction, DB write. Unit tests with synthetic point data.

### Commit 7 — `feat: qa agent and tools`
agents.py (qa) — qa_agent with all tools: DB read, distribution checks, fold balance, target range. Unit tests.

### Commit 8 — `feat: orchestrator agent and tools`
agents.py (orchestrator) + orchestrator.py — orchestrator_agent, main pipeline loop, replan logic, abort condition. Tools: run_ingestor, run_transformer, run_qa, replan, abort.

### Commit 9 — `feat: runners`
runners.py — thin async runner per agent: builds deps from state, calls agent.run(), writes result fields back to state, emits Logfire spans. Unit tests with all agents mocked.

### Commit 10 — `feat: entrypoint and CLI`
main.py — argparse for aoi_bbox + date_range, run_id generation via uuid, calls orchestrator runner, prints formatted decision_log on exit

### Commit 11 — `feat: integration tests`
tests/integration/ — full pipeline smoke test with all agents mocked at agent.run level, real SQLite via tmp_path fixture, assert correct DB row counts and state fields after happy path and replan path

---

## Key Design Decisions to Preserve

- **Agents own their I/O and computation via tools.** The ingestor agent queries Earth Engine via a registered @agent.tool. The transformer writes to SQLite via a tool. Pure computation (variogram, filter math, statistics) also lives in registered tools — not in runner code. Use @agent.tool for everything possible.
- **Runners are thin.** A runner's only job is: build deps from state → call agent.run() → write result back to state → emit Logfire spans. No logic lives in runners that could be a tool.
- **PipelineState is the single source of truth.** Agents receive deps (ephemeral per-call context), never the state object directly. State accumulates across the full pipeline.
- **Logfire traces cover everything.** PydanticAI agent calls are auto-instrumented. Wrap non-agent steps in logfire.span() manually so the full pipeline is visible in one trace.
- **All agent outputs are validated Pydantic BaseModels.** No JSON parsing, no KeyErrors, no string matching on model output.
- **recommended_action is always an explicit string literal** ("proceed", "widen_date_range", "escalate" etc.) so the orchestrator branches deterministically without further model calls.
- **pytest-asyncio with asyncio_mode = auto.** No @pytest.mark.asyncio decorator needed on individual tests.
- **Mock at agent.run in all unit tests.** Never call the real model in tests. Integration tests mock at the same level and use a real SQLite DB via tmp_path.