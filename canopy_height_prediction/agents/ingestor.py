from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from canopy_height_prediction._helpers import _fc_to_gdf, _load_gedi_shots
from canopy_height_prediction._models import (
    GediQueryResult,
    IngestorDecision,
    IngestorDeps,
    Sentinel2QueryResult,
)

load_dotenv()


INGESTOR_SYSTEM_PROMPT = """
You are the Ingestor agent in a canopy height estimation pipeline.

Your job: query GEDI L2A shot data and Sentinel-2 imagery statistics over the
given AOI and date range, reason about data quality, choose filter thresholds,
and either commit the filtered shots to the database or escalate.

Tool call sequence:
1. Call query_gedi_earthengine — examine rh98, sensitivity, and slope histograms.
2. Call query_sentinel2 — check scene count and cloud cover.
3. Reason about thresholds. Do NOT just use the suggested defaults:
   - If 80%+ of shots exceed the suggested sensitivity threshold, consider relaxing it.
   - If high-slope shots show unusually high rh98 variance, tighten slope_max_deg.
   - If replan_count > 0, soften sensitivity by ~0.05 and slope by ~5°.
4. Call compute_shot_density with your chosen thresholds.
5. If shots_per_km2 < min_shot_density_per_km2:
   - Set passed=False, recommended_action="replan_widen_date".
   - Do NOT call apply_quality_filters or write_raw_shots_to_db.
6. Otherwise call apply_quality_filters then write_raw_shots_to_db.

Rationale must be specific, e.g.:
"Accepted 4,200/6,800 shots — rejected 38% sensitivity < 0.95, 12% slope > 28°, 8% no S2 match."
Never write "rejected poor quality shots."
""".strip()

ingestor_agent: Agent[IngestorDeps, IngestorDecision] = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=IngestorDeps,
    output_type=IngestorDecision,
    system_prompt=INGESTOR_SYSTEM_PROMPT,
)


@ingestor_agent.tool
def query_gedi_earthengine(ctx: RunContext[IngestorDeps]) -> dict:
    """
    Query LARSE/GEDI/GEDI02_A_002 (vector shots) over aoi_bbox for the date range.
    Returns summary statistics only — no data download.
    Includes: total_shots, histograms for rh98/sensitivity/slope (10 bins each),
    quality_flag_counts, raw_shot_density_per_km2.
    Call this first to decide appropriate filter thresholds.
    """
    import ee

    deps = ctx.deps
    min_lon, min_lat, max_lon, max_lat = deps.aoi_bbox
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    collection = _load_gedi_shots(aoi, deps.date_start, deps.date_end)
    total = collection.size().getInfo()

    def _histogram(field: str, min_val: float, max_val: float) -> list:
        # Call .getInfo() on the whole reduceColumns dict, then extract in Python.
        # Chaining .get("histogram").getInfo() returns None for merged collections.
        result = collection.reduceColumns(
            ee.Reducer.fixedHistogram(min_val, max_val, 10),
            [field],
        ).getInfo()
        hist = (result or {}).get("histogram")
        if not hist:
            return []
        return [[round(row[0], 3), int(row[1])] for row in hist]

    rh98_hist = _histogram("rh98", 0, 60)
    sensitivity_hist = _histogram("sensitivity", 0, 1)
    slope_hist = _histogram("slope", 0, 60)

    quality_counts = (
        collection.aggregate_histogram("quality_flag").getInfo()
    )

    aoi_area_km2 = aoi.area().divide(1e6).getInfo()
    density = total / aoi_area_km2 if aoi_area_km2 > 0 else 0.0

    return GediQueryResult(
        total_shots=total,
        aoi_area_km2=round(aoi_area_km2, 2),
        raw_shot_density_per_km2=round(density, 4),
        rh98_histogram=rh98_hist,
        sensitivity_histogram=sensitivity_hist,
        slope_histogram=slope_hist,
        quality_flag_counts=quality_counts,
    ).model_dump()


@ingestor_agent.tool
def query_sentinel2(ctx: RunContext[IngestorDeps]) -> dict:
    """
    Query COPERNICUS/S2_SR_HARMONIZED over aoi_bbox for the date range.
    Applies SCL cloud mask (excludes SCL 3, 8, 9, 10).
    Returns: scene_count, mean_cloud_cover_pct, temporal_coverage_days, band_availability.
    """
    import ee

    deps = ctx.deps
    min_lon, min_lat, max_lon, max_lat = deps.aoi_bbox
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(deps.date_start, deps.date_end)
    )

    scene_count = collection.size().getInfo()

    def _cloud_pct(image):
        scl = image.select("SCL")
        cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
        cloud_pct = cloud_mask.reduceRegion(
            reducer=ee.Reducer.mean(), geometry=aoi, scale=100, maxPixels=1e8
        ).get("SCL")
        return image.set("cloud_pct", cloud_pct)

    cloud_stats = (
        collection.map(_cloud_pct)
        .aggregate_stats("cloud_pct")
        .getInfo()
    )
    mean_cloud_pct = round((cloud_stats.get("mean", 0) or 0) * 100, 1)

    return Sentinel2QueryResult(
        scene_count=scene_count,
        mean_cloud_cover_pct=mean_cloud_pct,
        temporal_coverage_days=None,  # date arithmetic handled client-side
        bands_available=["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "SCL"],
    ).model_dump()


@ingestor_agent.tool
def compute_shot_density(
    ctx: RunContext[IngestorDeps],
    sensitivity_min: float,
    slope_max_deg: float,
) -> dict:
    """
    Apply the given thresholds to GEDI shots and return density metrics WITHOUT
    writing to the database. Use this to evaluate whether filtered density meets
    min_shot_density_per_km2 before committing.
    Returns: accepted_shots, rejected_sensitivity, rejected_slope, rejected_quality_flag,
    shots_per_km2, spatial_coverage_pct.
    """
    import ee

    deps = ctx.deps
    min_lon, min_lat, max_lon, max_lat = deps.aoi_bbox
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    base = _load_gedi_shots(aoi, deps.date_start, deps.date_end)
    total = base.size().getInfo()

    after_sensitivity = base.filter(ee.Filter.gte("sensitivity", sensitivity_min))
    after_slope = after_sensitivity.filter(ee.Filter.lte("slope", slope_max_deg))
    after_quality = after_slope.filter(ee.Filter.eq("quality_flag", 1))

    n_sensitivity = total - after_sensitivity.size().getInfo()
    n_slope = after_sensitivity.size().getInfo() - after_slope.size().getInfo()
    n_quality = after_slope.size().getInfo() - after_quality.size().getInfo()
    accepted = after_quality.size().getInfo()

    aoi_area_km2 = aoi.area().divide(1e6).getInfo()
    density = accepted / aoi_area_km2 if aoi_area_km2 > 0 else 0.0

    return {
        "total_shots": total,
        "accepted_shots": accepted,
        "rejected_sensitivity": n_sensitivity,
        "rejected_slope": n_slope,
        "rejected_quality_flag": n_quality,
        "shots_per_km2": round(density, 4),
        "spatial_coverage_pct": None,  # computed after materialization
    }


@ingestor_agent.tool
def apply_quality_filters(
    ctx: RunContext[IngestorDeps],
    sensitivity_min: float,
    slope_max_deg: float,
) -> dict:
    """
    Apply the chosen thresholds to GEDI shots and materialize the accepted shots
    as a DataFrame held in deps._filtered_shots. Call only after validating density
    with compute_shot_density. Call write_raw_shots_to_db afterward to persist.
    Returns: accepted_shots, per-filter rejection counts.
    """
    import ee

    deps = ctx.deps
    min_lon, min_lat, max_lon, max_lat = deps.aoi_bbox
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    base = _load_gedi_shots(aoi, deps.date_start, deps.date_end)
    total = base.size().getInfo()

    filtered = (
        base.filter(ee.Filter.gte("sensitivity", sensitivity_min))
        .filter(ee.Filter.lte("slope", slope_max_deg))
        .filter(ee.Filter.eq("quality_flag", 1))
    )
    accepted = filtered.size().getInfo()

    gdf = _fc_to_gdf(
        filtered.select(["shot_number", "rh98", "sensitivity", "slope", "beam", "quality_flag"])
    )
    deps._filtered_shots = gdf

    return {
        "total_shots": total,
        "accepted_shots": accepted,
        "rejection_rate_pct": round((1 - accepted / total) * 100, 1) if total > 0 else 0,
        "rows_in_memory": len(gdf),
    }


@ingestor_agent.tool
def write_raw_shots_to_db(ctx: RunContext[IngestorDeps]) -> dict:
    """
    Write the materialized accepted shots from apply_quality_filters to the
    gedi_shots_raw table in SQLite. Must be called after apply_quality_filters.
    Returns: rows_written.
    """
    from canopy_height_prediction.db import GediShotRaw, get_session

    deps = ctx.deps
    gdf = deps._filtered_shots
    if gdf is None:
        return {"error": "No filtered shots in memory — call apply_quality_filters first."}

    rows = []
    for _, row in gdf.iterrows():
        geom = row.get("geometry")
        lat = geom.y if geom else None
        lon = geom.x if geom else None
        rows.append(
            GediShotRaw(
                run_id=deps.run_id,
                shot_id=str(row.get("shot_number", "")),
                lat=lat,
                lon=lon,
                rh98=row.get("rh98"),
                sensitivity=row.get("sensitivity"),
                slope=row.get("slope"),
                beam=row.get("beam"),
                quality_flag=row.get("quality_flag"),
            )
        )

    with get_session() as session:
        session.add_all(rows)
        session.commit()

    return {"rows_written": len(rows)}
