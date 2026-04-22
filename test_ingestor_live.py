"""
Manual smoke test for the Ingestor agent.

Run once to authenticate GEE:
    uv run python -c "import ee; ee.Authenticate()"

Then run this script:
    uv run python test_ingestor_live.py
"""

import asyncio
import uuid

import ee

from canopy_height_prediction.agents import IngestorDeps, ingestor_agent
from canopy_height_prediction.db import GediShotRaw, PipelineRun, get_session, init_db

# Small AOI: ~100km x 100km slice of northern California forests (Tahoe area)
AOI_BBOX = (-120.5, 38.5, -119.5, 39.5)
DATE_START = "2023-01-01"
DATE_END = "2023-04-01"  # GEDI decommissioned early 2024; use historical data


async def main():
    # 1. Init DB tables
    init_db()

    # 2. Create a pipeline_run row (required by FK on gedi_shots_raw)
    run_id = str(uuid.uuid4())[:8]
    with get_session() as session:
        session.add(PipelineRun(
            run_id=run_id,
            aoi_bbox={"bbox": list(AOI_BBOX)},
            date_range={"start": DATE_START, "end": DATE_END},
            status="running",
        ))
        session.commit()
    print(f"run_id: {run_id}")

    # 3. Authenticate + initialize GEE
    ee.Initialize(project="canopy-height-ml")

    # 4. Build deps and run the agent
    deps = IngestorDeps(
        run_id=run_id,
        aoi_bbox=AOI_BBOX,
        date_start=DATE_START,
        date_end=DATE_END,
        replan_count=0,
    )

    print("Running ingestor agent...")
    result = await ingestor_agent.run(
        "Ingest GEDI L2A and Sentinel-2 data for the given AOI and date range.",
        deps=deps,
    )

    decision = result.output
    print("\n--- Decision ---")
    print(f"passed:             {decision.passed}")
    print(f"sensitivity_min:    {decision.sensitivity_min}")
    print(f"slope_max_deg:      {decision.slope_max_deg}")
    print(f"raw_shots:          {decision.raw_shots}")
    print(f"accepted_shots:     {decision.accepted_shots}")
    print(f"recommended_action: {decision.recommended_action}")
    print(f"\nRationale:\n{decision.rationale}")
    if decision.warnings:
        print(f"\nWarnings: {decision.warnings}")

    # 5. Check DB rows written
    with get_session() as session:
        count = session.query(GediShotRaw).filter_by(run_id=run_id).count()
    print(f"\nRows in gedi_shots_raw: {count}")


if __name__ == "__main__":
    asyncio.run(main())
