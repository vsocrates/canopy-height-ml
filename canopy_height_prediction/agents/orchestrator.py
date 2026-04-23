from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from canopy_height_prediction._models import (
    IngestorDeps,
    OrchestratorDecision,
    OrchestratorDeps,
    QADeps,
    TransformerDeps,
)

load_dotenv()
# Import siblings directly — NOT via agents/__init__ to avoid circular imports.
from canopy_height_prediction.agents.ingestor import ingestor_agent
from canopy_height_prediction.agents.qa import qa_agent
from canopy_height_prediction.agents.transformer import transformer_agent


ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Orchestrator agent in a canopy height estimation pipeline.

You own the full pipeline loop. Your job is to run each stage in order, evaluate
its decision, and either proceed to the next stage, replan with adjusted parameters,
or abort if the data is fundamentally unusable.

Pipeline stages and tool call sequence:
1. Call run_ingestor — queries GEDI + Sentinel-2, filters shots, writes raw shots to DB.
2. If ingestor passed: call run_transformer — extracts S2 bands, assigns spatial folds.
3. If transformer passed: call run_qa — validates the cleaned dataset.
4. If QA passed: return action="proceed" with the final parameters.

On failure (any agent returns passed=False):
- Check recommended_action from the failed agent:
  * "replan_widen_date": call replan with extend_days=90 to expand the date window.
  * "replan_relax_thresholds": call replan with relax_thresholds=True to soften
    sensitivity_min (−0.05) and slope_max_deg (+5°).
  * "abort": call abort immediately — do not replan.
- After replan, re-run only the failed stage (and all subsequent stages).
  Do NOT restart from the beginning unless the ingestor itself failed.
- If replan_count reaches max_replans, call abort.

Rationale must summarize what each stage returned and why you chose to proceed,
replan, or abort, e.g.:
"Ingestor: 3,200 shots accepted (passed). Transformer: nan_rate=4%, block=12km (passed).
QA: rh98 range [1.2, 55m], folds balanced — no issues. Proceeding."
""".strip()


orchestrator_agent: Agent[OrchestratorDeps, OrchestratorDecision] = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=OrchestratorDeps,
    output_type=OrchestratorDecision,
    system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
)


@orchestrator_agent.tool
async def run_ingestor(ctx: RunContext[OrchestratorDeps]) -> dict:
    """
    Run the Ingestor agent for the current AOI, date range, and quality thresholds.
    Creates the pipeline_run DB record if it does not yet exist.
    Returns the IngestorDecision fields so you can evaluate pass/fail.
    """
    from canopy_height_prediction.db import PipelineRun, get_session

    deps = ctx.deps

    with get_session() as session:
        if not session.get(PipelineRun, deps.run_id):
            session.add(PipelineRun(
                run_id=deps.run_id,
                aoi_bbox={"bbox": list(deps.aoi_bbox)},
                date_range={"start": deps.date_start, "end": deps.date_end},
                status="running",
            ))
            session.commit()

    ingestor_deps = IngestorDeps(
        run_id=deps.run_id,
        aoi_bbox=deps.aoi_bbox,
        date_start=deps.date_start,
        date_end=deps.date_end,
        replan_count=deps.replan_count,
        sensitivity_min_suggested=deps.sensitivity_min,
        slope_max_deg_suggested=deps.slope_max_deg,
        min_shot_density_per_km2=deps.min_shot_density_per_km2,
        gee_project=deps.gee_project,
    )

    result = await ingestor_agent.run(
        "Ingest GEDI L2A and Sentinel-2 data for the given AOI and date range.",
        deps=ingestor_deps,
    )
    d = result.output
    if deps.state is not None:
        deps.state.raw_shots = d.raw_shots
        deps.state.accepted_shots = d.accepted_shots
        deps.state.ingestor_passed = d.passed
        deps.state.ingestor_rationale = d.rationale
        deps.state.ingestor_recommended_action = d.recommended_action
    return {
        "passed": d.passed,
        "sensitivity_min": d.sensitivity_min,
        "slope_max_deg": d.slope_max_deg,
        "raw_shots": d.raw_shots,
        "accepted_shots": d.accepted_shots,
        "recommended_action": d.recommended_action,
        "rationale": d.rationale,
        "warnings": d.warnings,
    }


@orchestrator_agent.tool
async def run_transformer(ctx: RunContext[OrchestratorDeps]) -> dict:
    """
    Run the Transformer agent to match GEDI shots to Sentinel-2 bands, compute
    spatial CV blocks, assign folds, and write the cleaned dataset to the DB.
    Returns the TransformerDecision fields so you can evaluate pass/fail.
    """
    deps = ctx.deps

    transformer_deps = TransformerDeps(
        run_id=deps.run_id,
        aoi_bbox=deps.aoi_bbox,
        date_start=deps.date_start,
        date_end=deps.date_end,
        replan_count=deps.replan_count,
        n_folds=deps.n_folds,
        gee_project=deps.gee_project,
    )

    result = await transformer_agent.run(
        "Match GEDI shots to Sentinel-2 bands, assign spatial CV folds, and write cleaned data.",
        deps=transformer_deps,
    )
    d = result.output
    if deps.state is not None:
        deps.state.cv_block_size_km = d.cv_block_size_km
        deps.state.n_folds = d.n_folds
        deps.state.transformer_passed = d.passed
        deps.state.transformer_rationale = d.rationale
        deps.state.transformer_recommended_action = d.recommended_action
    return {
        "passed": d.passed,
        "cv_block_size_km": d.cv_block_size_km,
        "n_folds": d.n_folds,
        "recommended_action": d.recommended_action,
        "rationale": d.rationale,
        "warnings": d.warnings,
    }


@orchestrator_agent.tool
async def run_qa(ctx: RunContext[OrchestratorDeps]) -> dict:
    """
    Run the QA agent to validate the cleaned dataset: feature distributions,
    fold balance, and rh98 target range.
    Returns the QADecision fields so you can evaluate pass/fail.
    """
    deps = ctx.deps

    qa_deps = QADeps(
        run_id=deps.run_id,
        replan_count=deps.replan_count,
    )

    result = await qa_agent.run(
        "Validate the cleaned dataset for model training readiness.",
        deps=qa_deps,
    )
    d = result.output
    if deps.state is not None:
        deps.state.qa_passed = d.passed
        deps.state.qa_rationale = d.rationale
        deps.state.qa_issues = d.issues
        deps.state.qa_recommended_action = d.recommended_action
    return {
        "passed": d.passed,
        "issues": d.issues,
        "recommended_action": d.recommended_action,
        "rationale": d.rationale,
    }


@orchestrator_agent.tool
def replan(
    ctx: RunContext[OrchestratorDeps],
    reason: str,
    extend_days: int = 0,
    relax_thresholds: bool = False,
) -> dict:
    """
    Update pipeline parameters in preparation for re-running a failed stage.
    - extend_days: extend date_end forward by this many days (use for replan_widen_date).
    - relax_thresholds: if True, reduce sensitivity_min by 0.05 and increase
      slope_max_deg by 5° (use for replan_relax_thresholds).
    Increments replan_count. Returns the updated parameters and replan_count.
    Call abort instead if replan_count has already reached max_replans.
    """
    from datetime import datetime, timedelta

    deps = ctx.deps

    if deps.replan_count >= deps.max_replans:
        return {
            "error": f"max_replans={deps.max_replans} already reached — call abort instead.",
        }

    deps.replan_count += 1

    if extend_days > 0:
        new_end = datetime.strptime(deps.date_end, "%Y-%m-%d") + timedelta(days=extend_days)
        deps.date_end = new_end.strftime("%Y-%m-%d")

    if relax_thresholds:
        deps.sensitivity_min = round(max(deps.sensitivity_min - 0.05, 0.70), 2)
        deps.slope_max_deg = round(min(deps.slope_max_deg + 5.0, 45.0), 1)

    return {
        "replan_count": deps.replan_count,
        "max_replans": deps.max_replans,
        "reason": reason,
        "date_start": deps.date_start,
        "date_end": deps.date_end,
        "sensitivity_min": deps.sensitivity_min,
        "slope_max_deg": deps.slope_max_deg,
    }


@orchestrator_agent.tool
def abort(ctx: RunContext[OrchestratorDeps], reason: str) -> dict:
    """
    Record that the pipeline is being aborted and return the reason.
    Use when a stage returns recommended_action="abort", or when replan_count
    has reached max_replans without success.
    """
    deps = ctx.deps
    return {
        "aborted": True,
        "reason": reason,
        "replan_count": deps.replan_count,
        "date_start": deps.date_start,
        "date_end": deps.date_end,
        "sensitivity_min": deps.sensitivity_min,
        "slope_max_deg": deps.slope_max_deg,
    }
