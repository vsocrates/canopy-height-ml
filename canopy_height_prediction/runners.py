"""
Thin async runners — one per agent.

Each runner's only job:
  1. Build the agent's Deps from PipelineState.
  2. Call agent.run().
  3. Write scalar results back to PipelineState.
  4. Emit a Logfire span so every stage appears in the same trace.

No logic, no EE calls, no DB writes — those all live in agent tools.
"""

import logfire

from canopy_height_prediction.agents import (
    IngestorDeps,
    OrchestratorDeps,
    QADeps,
    TransformerDeps,
    ingestor_agent,
    orchestrator_agent,
    qa_agent,
    transformer_agent,
)
from canopy_height_prediction.state import PipelineState


async def run_ingestor(state: PipelineState) -> PipelineState:
    logfire.event("agent.ingestor.start", run_id=state.run_id)
    with logfire.span("runner.ingestor", run_id=state.run_id):
        deps = IngestorDeps(
            run_id=state.run_id,
            aoi_bbox=state.aoi_bbox,
            date_start=state.date_range[0],
            date_end=state.date_range[1],
            replan_count=state.replan_count,
        )
        result = await ingestor_agent.run(
            "Ingest GEDI L2A and Sentinel-2 data for the given AOI and date range.",
            deps=deps,
        )
        d = result.output
        state.raw_shots = d.raw_shots
        state.accepted_shots = d.accepted_shots
        state.ingestor_rationale = d.rationale
        state.ingestor_passed = d.passed
        state.ingestor_recommended_action = d.recommended_action
        state.decision_log.append({"agent": "ingestor", **d.model_dump()})
    return state


async def run_transformer(state: PipelineState) -> PipelineState:
    logfire.event("agent.transformer.start", run_id=state.run_id)
    with logfire.span("runner.transformer", run_id=state.run_id):
        deps = TransformerDeps(
            run_id=state.run_id,
            aoi_bbox=state.aoi_bbox,
            date_start=state.date_range[0],
            date_end=state.date_range[1],
            replan_count=state.replan_count,
            n_folds=state.n_folds,
        )
        result = await transformer_agent.run(
            "Match GEDI shots to Sentinel-2 bands, assign spatial CV folds, and write cleaned data.",
            deps=deps,
        )
        d = result.output
        state.cv_block_size_km = d.cv_block_size_km
        state.n_folds = d.n_folds
        state.transformer_rationale = d.rationale
        state.transformer_passed = d.passed
        state.transformer_recommended_action = d.recommended_action
        state.decision_log.append({"agent": "transformer", **d.model_dump()})
    return state


async def run_qa(state: PipelineState) -> PipelineState:
    logfire.event("agent.qa.start", run_id=state.run_id)
    with logfire.span("runner.qa", run_id=state.run_id):
        deps = QADeps(
            run_id=state.run_id,
            replan_count=state.replan_count,
        )
        result = await qa_agent.run(
            "Validate the cleaned dataset for model training readiness.",
            deps=deps,
        )
        d = result.output
        state.qa_rationale = d.rationale
        state.qa_passed = d.passed
        state.qa_issues = d.issues
        state.qa_recommended_action = d.recommended_action
        state.decision_log.append({"agent": "qa", **d.model_dump()})
    return state


async def run_orchestrator(state: PipelineState) -> PipelineState:
    logfire.event("agent.orchestrator.start", run_id=state.run_id)
    with logfire.span("runner.orchestrator", run_id=state.run_id):
        deps = OrchestratorDeps(
            run_id=state.run_id,
            aoi_bbox=state.aoi_bbox,
            date_start=state.date_range[0],
            date_end=state.date_range[1],
            replan_count=state.replan_count,
        )
        result = await orchestrator_agent.run(
            "Run the full canopy height estimation pipeline: ingest, transform, and validate.",
            deps=deps,
        )
        d = result.output
        state.replan_count = d.replan_count
        # Write back any parameter updates the orchestrator settled on
        if d.final_date_start:
            state.date_range = (d.final_date_start, state.date_range[1])
        if d.final_date_end:
            state.date_range = (state.date_range[0], d.final_date_end)
        state.decision_log.append({"agent": "orchestrator", **d.model_dump()})
    return state
