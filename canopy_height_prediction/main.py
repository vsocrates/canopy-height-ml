"""
CLI entrypoint for the canopy height estimation pipeline.

Usage:
    uv run python -m canopy_height_prediction.main \
        --bbox=-122.5,37.5,-121.5,38.5 \
        --date-start=2023-01-01 \
        --date-end=2023-06-01

The orchestrator agent drives ingest → transform → QA with replan logic.
"""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import logfire

from canopy_height_prediction.runners import run_orchestrator
from canopy_height_prediction.state import PipelineState


def _parse_bbox(s: str) -> tuple[float, float, float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be four floats: min_lon,min_lat,max_lon,max_lat")
    return tuple(parts)


def _print_summary(state: PipelineState) -> None:
    print("\n" + "=" * 60)
    print(f"Run ID : {state.run_id}")
    print(f"AOI    : {state.aoi_bbox}")
    print(f"Dates  : {state.date_range[0]} → {state.date_range[1]}")
    print("-" * 60)
    print(f"Ingestor  : {'PASS' if state.ingestor_passed else 'FAIL'}  "
          f"({state.accepted_shots}/{state.raw_shots} shots accepted)")
    print(f"Transformer: {'PASS' if state.transformer_passed else 'FAIL'}  "
          f"(block={state.cv_block_size_km} km, folds={state.n_folds})")
    print(f"QA        : {'PASS' if state.qa_passed else 'FAIL'}  "
          f"issues={state.qa_issues}")
    print(f"Replans   : {state.replan_count}")
    print("-" * 60)
    print("Decision log:")
    for entry in state.decision_log:
        agent = entry.get("agent", "?")
        rationale = entry.get("rationale", "")
        print(f"  [{agent}] {rationale}")
    print("=" * 60)


async def _run(args: argparse.Namespace) -> None:
    run_id = args.run_id or f"run-{uuid.uuid4().hex[:8]}"
    state = PipelineState(
        run_id=run_id,
        aoi_bbox=args.bbox,
        date_range=(args.date_start, args.date_end),
        n_folds=args.n_folds,
    )

    logfire.configure(send_to_logfire=False)

    print(f"Starting pipeline  run_id={run_id}")
    state = await run_orchestrator(state)

    _print_summary(state)

    if args.save_figs and state.qa_passed:
        from canopy_height_prediction.plots import generate_pipeline_figures
        fig_paths = generate_pipeline_figures(state.run_id, Path(args.save_figs))
        for p in fig_paths:
            print(f"Figure: {p}")

    if args.output:
        with open(args.output, "w") as fh:
            json.dump(
                {
                    "run_id": state.run_id,
                    "aoi_bbox": list(state.aoi_bbox),
                    "date_range": list(state.date_range),
                    "ingestor_passed": state.ingestor_passed,
                    "transformer_passed": state.transformer_passed,
                    "qa_passed": state.qa_passed,
                    "replan_count": state.replan_count,
                    "decision_log": state.decision_log,
                },
                fh,
                indent=2,
            )
        print(f"Results written to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canopy height estimation pipeline (GEDI + Sentinel-2 → XGBoost)",
    )
    parser.add_argument(
        "--bbox",
        required=True,
        type=_parse_bbox,
        metavar="min_lon,min_lat,max_lon,max_lat",
        help="Area of interest bounding box",
    )
    parser.add_argument("--date-start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--date-end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--run-id", default=None, help="Optional run ID (auto-generated if omitted)")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of spatial CV folds (default: 5)")
    parser.add_argument("--output", default=None, metavar="PATH", help="Write JSON summary to this file")
    parser.add_argument("--save-figs", default=None, metavar="DIR",
                        help="Write diagnostic PNG figures to this directory (only when QA passes)")

    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
