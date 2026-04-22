"""
Manual smoke test for the Transformer agent.

Requires the ingestor to have already run and written rows to gedi_shots_raw
for the given run_id, OR pass an existing run_id via RUN_ID env var.

Run once to authenticate GEE:
    uv run python -c "import ee; ee.Authenticate()"

Then run the ingestor first (to populate gedi_shots_raw), or set RUN_ID to
an existing run that already has raw shots:
    RUN_ID=<existing-run-id> uv run python test_transformer_live.py

To run end-to-end from scratch (ingestor → transformer in sequence):
    uv run python test_ingestor_live.py   # note the run_id it prints
    RUN_ID=<that-run-id> uv run python test_transformer_live.py
"""

import asyncio
import os
import uuid

import ee

from canopy_height_prediction.agents import TransformerDeps, transformer_agent
from canopy_height_prediction.db import GediShotCleaned, GediShotRaw, PipelineRun, get_session, init_db

AOI_BBOX = (-120.4, 39.2, -120.2, 39.4)
DATE_START = "2023-01-01"
DATE_END = "2023-04-01"


async def main():
    init_db()

    run_id = os.getenv("RUN_ID")
    if run_id:
        with get_session() as session:
            raw_count = session.query(GediShotRaw).filter_by(run_id=run_id).count()
        if raw_count == 0:
            print(f"ERROR: run_id={run_id} has no rows in gedi_shots_raw.")
            print("Run test_ingestor_live.py first, or set RUN_ID to a valid run.")
            return
        print(f"Using existing run_id: {run_id} ({raw_count} raw shots)")
    else:
        run_id = str(uuid.uuid4())[:8]
        with get_session() as session:
            session.add(PipelineRun(
                run_id=run_id,
                aoi_bbox={"bbox": list(AOI_BBOX)},
                date_range={"start": DATE_START, "end": DATE_END},
                status="running",
            ))
            # Seed a few synthetic raw shots so the transformer has something to work with
            for i in range(10):
                session.add(GediShotRaw(
                    run_id=run_id,
                    shot_id=f"synthetic_{i}",
                    lat=39.21 + i * 0.018,
                    lon=-120.39 + i * 0.018,
                    rh98=10.0 + i * 2,
                    sensitivity=0.97,
                    slope=8.0,
                    beam="BEAM0000",
                    quality_flag=1,
                ))
            session.commit()
        print(f"Created new run_id: {run_id} (seeded 10 synthetic shots)")

    ee.Initialize(project="canopy-height-ml")

    deps = TransformerDeps(
        run_id=run_id,
        aoi_bbox=AOI_BBOX,
        date_start=DATE_START,
        date_end=DATE_END,
        replan_count=0,
        n_folds=5,
    )

    print("Running transformer agent...")
    result = await transformer_agent.run(
        "Match GEDI shots to Sentinel-2 bands, assign spatial CV folds, and write cleaned data.",
        deps=deps,
    )

    decision = result.output
    print("\n--- Decision ---")
    print(f"passed:             {decision.passed}")
    print(f"cv_block_size_km:   {decision.cv_block_size_km}")
    print(f"n_folds:            {decision.n_folds}")
    print(f"recommended_action: {decision.recommended_action}")
    print(f"\nRationale:\n{decision.rationale}")
    if decision.warnings:
        print(f"\nWarnings: {decision.warnings}")

    with get_session() as session:
        count = session.query(GediShotCleaned).filter_by(run_id=run_id).count()
        sample = session.query(GediShotCleaned).filter_by(run_id=run_id).first()
    print(f"\nRows in gedi_shots_cleaned: {count}")
    if sample:
        print(f"Sample row — shot_id={sample.shot_id}, ndvi={sample.ndvi:.3f}, "
              f"scl_valid_obs={sample.scl_valid_obs}, fold={sample.fold}, split={sample.split}")


if __name__ == "__main__":
    asyncio.run(main())
