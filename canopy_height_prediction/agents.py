import os

import logfire
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from canopy_height_prediction._helpers import (
    MIN_VALID_OBS,
    _estimate_variogram_range,
    _fc_to_gdf,
    _load_gedi_shots,
)
from canopy_height_prediction._models import (
    GediQueryResult,
    IngestorDecision,
    IngestorDeps,
    OrchestratorDecision,
    OrchestratorDeps,
    QADecision,
    QADeps,
    Sentinel2QueryResult,
    TransformerDecision,
    TransformerDeps,
)

load_dotenv()

if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure()
    logfire.instrument_pydantic_ai()


# ---------------------------------------------------------------------------
# Ingestor Agent + Tools
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# QA Agent + Tools
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = """
You are the QA agent in a canopy height estimation pipeline.

Your job: read the cleaned dataset from the database, inspect it for issues that
would invalidate model training, and return a pass/fail decision with specific
rationale and an issues list.

Tool call sequence:
1. Call read_cleaned_shots_from_db — loads the cleaned dataset into memory.
   If rows=0, immediately set passed=False, recommended_action="replan_widen_date".
2. Call check_feature_distributions — inspect band stats and NaN rates.
   Flag any band with >10% NaN as a potential leakage or coverage issue.
3. Call check_fold_balance — inspect fold sizes.
   Flag if any fold has <5% or >40% of total shots (severe imbalance).
4. Call check_target_range — inspect rh98 distribution.
   Flag if >5% of shots have rh98 <= 0 (physically impossible for vegetated areas).
   Flag if >10% of shots have rh98 > 60m (implausible for most forest types).

Synthesize all findings into a single pass/fail decision:
- passed=True only if no critical issues were found across all checks.
- Use recommended_action="replan_widen_date" if the core problem is data scarcity.
- Use recommended_action="replan_relax_thresholds" if the problem is over-filtering.
- Use recommended_action="abort" only if the data is fundamentally corrupt or unusable.

Rationale must cite specific numbers, e.g.:
"rh98 range [0.1, 58.2]m — 0.8% negative values (flagged). Fold sizes balanced
(min 18%, max 24%). B11 NaN rate 2.1%. No critical issues — proceed."
Never write "data looks good."
""".strip()


qa_agent: Agent[QADeps, QADecision] = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=QADeps,
    output_type=QADecision,
    system_prompt=QA_SYSTEM_PROMPT,
)


@qa_agent.tool
def read_cleaned_shots_from_db(ctx: RunContext[QADeps]) -> dict:
    """
    Load all cleaned shots for this run from gedi_shots_cleaned into memory.
    Stores a DataFrame in deps._cleaned_df. Must be called first.
    Returns: rows, folds_present, split_counts, columns_available.
    """
    import pandas as pd

    from canopy_height_prediction.db import GediShotCleaned, get_session

    deps = ctx.deps
    with get_session() as session:
        db_rows = session.query(GediShotCleaned).filter_by(run_id=deps.run_id).all()

    if not db_rows:
        return {"rows": 0, "folds_present": [], "split_counts": {}, "columns_available": []}

    records = [
        {
            "shot_id": r.shot_id,
            "lat": r.lat, "lon": r.lon,
            "rh98": r.rh98,
            "b2": r.b2, "b3": r.b3, "b4": r.b4,
            "b8": r.b8, "b11": r.b11, "b12": r.b12,
            "ndvi": r.ndvi, "evi": r.evi,
            "scl_valid_obs": r.scl_valid_obs,
            "fold": r.fold,
            "split": r.split,
        }
        for r in db_rows
    ]
    df = pd.DataFrame(records)
    deps._cleaned_df = df

    folds = sorted(df["fold"].dropna().unique().tolist())
    split_counts = df["split"].value_counts().to_dict()
    return {
        "rows": len(df),
        "folds_present": [int(f) for f in folds],
        "split_counts": {str(k): int(v) for k, v in split_counts.items()},
        "columns_available": list(df.columns),
    }


@qa_agent.tool
def check_feature_distributions(ctx: RunContext[QADeps]) -> dict:
    """
    Inspect S2 band statistics and NaN rates in the cleaned dataset.
    Returns per-band mean, std, min, max, and nan_rate_pct.
    Bands with nan_rate_pct > 10% are flagged as potential coverage issues.
    Must be called after read_cleaned_shots_from_db.
    """
    deps = ctx.deps
    df = deps._cleaned_df
    if df is None:
        return {"error": "No data in memory — call read_cleaned_shots_from_db first."}

    band_cols = ["b2", "b3", "b4", "b8", "b11", "b12", "ndvi", "evi"]
    n = len(df)
    stats = {}
    flagged = []

    for col in band_cols:
        if col not in df.columns:
            continue
        series = df[col]
        nan_rate = round(series.isna().sum() / n * 100, 1)
        valid = series.dropna()
        stats[col] = {
            "mean": round(float(valid.mean()), 3) if len(valid) else None,
            "std": round(float(valid.std()), 3) if len(valid) > 1 else None,
            "min": round(float(valid.min()), 3) if len(valid) else None,
            "max": round(float(valid.max()), 3) if len(valid) else None,
            "nan_rate_pct": nan_rate,
        }
        if nan_rate > 10.0:
            flagged.append(f"{col}: nan_rate={nan_rate}%")

    return {"band_stats": stats, "flagged_bands": flagged}


@qa_agent.tool
def check_fold_balance(ctx: RunContext[QADeps]) -> dict:
    """
    Inspect the distribution of shots across spatial CV folds.
    Returns shots per fold and flags any fold with <5% or >40% of total shots.
    Must be called after read_cleaned_shots_from_db.
    """
    deps = ctx.deps
    df = deps._cleaned_df
    if df is None:
        return {"error": "No data in memory — call read_cleaned_shots_from_db first."}

    n = len(df)
    fold_counts = df["fold"].value_counts().sort_index()
    fold_pcts = (fold_counts / n * 100).round(1)

    flagged = []
    for fold, pct in fold_pcts.items():
        if pct < 5.0:
            flagged.append(f"fold {fold}: {pct}% of shots (under-represented)")
        elif pct > 40.0:
            flagged.append(f"fold {fold}: {pct}% of shots (over-represented)")

    return {
        "n_total": n,
        "fold_sizes": {str(int(k)): int(v) for k, v in fold_counts.items()},
        "fold_pcts": {str(int(k)): float(v) for k, v in fold_pcts.items()},
        "flagged_folds": flagged,
    }


@qa_agent.tool
def check_target_range(ctx: RunContext[QADeps]) -> dict:
    """
    Inspect the rh98 target variable distribution for physically implausible values.
    Flags: rh98 <= 0 (impossible for vegetated area), rh98 > 60m (implausible for most forests).
    Returns summary stats and flagged counts.
    Must be called after read_cleaned_shots_from_db.
    """
    deps = ctx.deps
    df = deps._cleaned_df
    if df is None:
        return {"error": "No data in memory — call read_cleaned_shots_from_db first."}

    rh98 = df["rh98"].dropna()
    n = len(df)
    n_valid = len(rh98)
    n_null = n - n_valid

    n_negative = int((rh98 <= 0).sum())
    n_implausible_high = int((rh98 > 60).sum())
    negative_pct = round(n_negative / n * 100, 1) if n > 0 else 0.0
    high_pct = round(n_implausible_high / n * 100, 1) if n > 0 else 0.0

    flagged = []
    if negative_pct > 5.0:
        flagged.append(f"{negative_pct}% of rh98 values are <= 0 (physically impossible)")
    if high_pct > 10.0:
        flagged.append(f"{high_pct}% of rh98 values exceed 60m (implausible for most forests)")
    if n_null > 0:
        flagged.append(f"{n_null} shots have null rh98 — target variable missing")

    return {
        "n_total": n,
        "n_valid_rh98": n_valid,
        "n_null_rh98": n_null,
        "rh98_min": round(float(rh98.min()), 2) if n_valid else None,
        "rh98_max": round(float(rh98.max()), 2) if n_valid else None,
        "rh98_mean": round(float(rh98.mean()), 2) if n_valid else None,
        "rh98_p5": round(float(rh98.quantile(0.05)), 2) if n_valid else None,
        "rh98_p95": round(float(rh98.quantile(0.95)), 2) if n_valid else None,
        "negative_pct": negative_pct,
        "implausible_high_pct": high_pct,
        "flagged": flagged,
    }


# ---------------------------------------------------------------------------
# Transformer Agent + Tools
# ---------------------------------------------------------------------------

TRANSFORMER_SYSTEM_PROMPT = """
You are the Transformer agent in a canopy height estimation pipeline.

Your job: match GEDI L2A shots to Sentinel-2 band values, estimate spatial
autocorrelation via variogram, choose a CV block size that exceeds the range,
assign spatial folds, and write the cleaned dataset to the database.

Tool call sequence:
1. Call extract_sentinel_bands — matches raw shots to S2 median bands and
   computes NDVI/EVI. Check both quality signals in the result:
   - If nan_rate_pct > 20%, set passed=False, recommended_action="replan_widen_date".
   - If valid_obs_below_threshold_pct > 50%, set passed=False, recommended_action="replan_widen_date"
     (too many shots have fewer than min_valid_obs_used clear-sky scenes — composite unreliable).
   - If valid_obs_below_threshold_pct > 20%, add a warning but continue.
   - Do NOT proceed to variogram if either failure condition is met.
2. Call compute_variogram — inspect range_km.
   The CV block size MUST exceed the variogram range to prevent spatial leakage.
3. Reason about block size:
   - Start with block_size_km = max(range_km * 1.5, 5.0).
   - If replan_count > 0, you may reduce toward range_km * 1.2 (minimum 3.0km).
4. Call generate_spatial_blocks with your chosen block_size_km.
5. Call assign_folds — uses the block assignments and deps.n_folds.
6. Call write_cleaned_shots_to_db.

Rationale must be specific, e.g.:
"Variogram range ~8km; chose block_size=12km (1.5× range). 4,100 matched shots
across 5 folds (fold sizes: [810, 825, 815, 830, 820]). nan_rate=3.2%."
Never write "assigned spatial folds" without citing block size and fold sizes.
""".strip()


transformer_agent: Agent[TransformerDeps, TransformerDecision] = Agent(
    "anthropic:claude-sonnet-4-6",
    deps_type=TransformerDeps,
    output_type=TransformerDecision,
    system_prompt=TRANSFORMER_SYSTEM_PROMPT,
)


@transformer_agent.tool
def extract_sentinel_bands(ctx: RunContext[TransformerDeps]) -> dict:
    """
    Load raw GEDI shots for this run from the DB, sample a Sentinel-2 median
    composite at each shot location (B2,B3,B4,B8,B11,B12), and compute NDVI/EVI.
    Stores the matched DataFrame in deps._matched_df.
    Returns: n_shots, n_matched, nan_rate_pct, band_preview (first 3 rows).
    """
    import ee
    import pandas as pd

    from canopy_height_prediction.db import GediShotRaw, get_session

    deps = ctx.deps

    with get_session() as session:
        rows = session.query(GediShotRaw).filter_by(run_id=deps.run_id).all()

    if not rows:
        return {"error": f"No raw shots found for run_id={deps.run_id}"}

    n_shots = len(rows)

    features = [
        ee.Feature(
            ee.Geometry.Point([r.lon, r.lat]),
            {"shot_id": r.shot_id, "rh98": r.rh98 or 0.0},
        )
        for r in rows
    ]
    fc = ee.FeatureCollection(features)

    min_lon, min_lat, max_lon, max_lat = deps.aoi_bbox
    aoi = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])

    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(deps.date_start, deps.date_end)
    )

    # Count scenes where SCL indicates clear land (4=veg, 5=bare soil, 6=water, 7=unclassified).
    # All other classes (0=no data, 1=saturated, 2=dark, 3=shadow, 8-11=cloud/snow) are invalid.
    valid_obs_count = (
        s2_col.select("SCL")
        .map(lambda img: img.remap([4, 5, 6, 7], [1, 1, 1, 1], 0).rename("valid_obs"))
        .sum()
        .rename("valid_obs_count")
    )

    s2 = s2_col.select(["B2", "B3", "B4", "B8", "B11", "B12"]).median()
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")
    evi = s2.expression(
        "2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1e-10)",
        {"B8": s2.select("B8"), "B4": s2.select("B4"), "B2": s2.select("B2")},
    ).rename("evi")
    image = s2.addBands([ndvi, evi, valid_obs_count])

    sampled = image.sampleRegions(collection=fc, scale=10, geometries=True)
    gdf = _fc_to_gdf(sampled)

    # Extract lat/lon from geometry, build clean DataFrame
    shot_lookup = {r.shot_id: (r.lat, r.lon) for r in rows}
    records = []
    band_cols = ["B2", "B3", "B4", "B8", "B11", "B12", "ndvi", "evi", "valid_obs_count"]
    for _, feat in gdf.iterrows():
        sid = str(feat.get("shot_id", ""))
        rh98 = feat.get("rh98")
        geom = feat.get("geometry")
        lat = geom.y if geom else (shot_lookup.get(sid, (None, None))[0])
        lon = geom.x if geom else (shot_lookup.get(sid, (None, None))[1])
        rec = {"shot_id": sid, "lat": lat, "lon": lon, "rh98": rh98}
        for b in band_cols:
            rec[b] = feat.get(b)
        records.append(rec)

    df = pd.DataFrame(records)
    n_matched = int(df[["B8", "B4"]].notna().all(axis=1).sum())
    nan_rate = round((1 - n_matched / n_shots) * 100, 1) if n_shots > 0 else 0.0

    below = int((df["valid_obs_count"].fillna(0) < MIN_VALID_OBS).sum())
    below_pct = round(below / n_shots * 100, 1) if n_shots > 0 else 0.0

    deps._matched_df = df

    preview = df[["shot_id", "rh98", "B8", "B4", "ndvi", "valid_obs_count"]].head(3).to_dict(
        orient="records"
    )
    return {
        "n_shots": n_shots,
        "n_matched": n_matched,
        "nan_rate_pct": nan_rate,
        "valid_obs_below_threshold_pct": below_pct,
        "min_valid_obs_used": MIN_VALID_OBS,
        "band_preview": preview,
    }


@transformer_agent.tool
def compute_variogram(ctx: RunContext[TransformerDeps]) -> dict:
    """
    Estimate the spatial autocorrelation range (km) of rh98 using a binned
    experimental variogram on the matched shots. Must be called after
    extract_sentinel_bands. Returns range_km and the variogram histogram.
    The CV block size should exceed this range to prevent spatial leakage.
    """
    deps = ctx.deps
    df = deps._matched_df
    if df is None:
        return {"error": "No matched shots — call extract_sentinel_bands first."}

    valid = df[["lat", "lon", "rh98"]].dropna()
    if len(valid) < 10:
        return {"range_km": 5.0, "note": "Too few valid shots for variogram; defaulting to 5km."}

    range_km, bin_centers, bin_sv = _estimate_variogram_range(
        valid["lat"].values, valid["lon"].values, valid["rh98"].values
    )
    return {
        "range_km": range_km,
        "n_points_used": len(valid),
        "variogram_lags": bin_centers,
        "variogram_semivariance": bin_sv,
    }


@transformer_agent.tool
def generate_spatial_blocks(
    ctx: RunContext[TransformerDeps],
    block_size_km: float,
) -> dict:
    """
    Assign each matched shot to a spatial grid block of block_size_km × block_size_km.
    Stores block_id on deps._matched_df (in-place). Must be called after
    compute_variogram. Returns n_blocks and shots-per-block statistics.
    """
    import numpy as np

    deps = ctx.deps
    df = deps._matched_df
    if df is None:
        return {"error": "No matched shots — call extract_sentinel_bands first."}

    min_lon, min_lat, max_lon, max_lat = deps.aoi_bbox
    mean_lat_rad = float(np.deg2rad((min_lat + max_lat) / 2))

    deg_lat_per_km = 1.0 / 111.0
    deg_lon_per_km = 1.0 / (111.0 * float(np.cos(mean_lat_rad)))

    block_deg_lat = block_size_km * deg_lat_per_km
    block_deg_lon = block_size_km * deg_lon_per_km

    block_x = ((df["lon"] - min_lon) / block_deg_lon).astype(int)
    block_y = ((df["lat"] - min_lat) / block_deg_lat).astype(int)
    n_blocks_x = max(int((max_lon - min_lon) / block_deg_lon) + 1, 1)
    df["block_id"] = block_y * n_blocks_x + block_x

    block_counts = df["block_id"].value_counts()
    return {
        "block_size_km": block_size_km,
        "n_blocks_occupied": int(block_counts.shape[0]),
        "shots_per_block_mean": round(float(block_counts.mean()), 1),
        "shots_per_block_min": int(block_counts.min()),
        "shots_per_block_max": int(block_counts.max()),
    }


@transformer_agent.tool
def assign_folds(ctx: RunContext[TransformerDeps]) -> dict:
    """
    Assign fold numbers (0..n_folds-1) to spatial blocks via round-robin, then
    mark fold 0 as 'test' and all others as 'train'. Stores the folded DataFrame
    in deps._folded_df. Must be called after generate_spatial_blocks.
    Returns fold sizes and split breakdown.
    """
    import pandas as pd

    deps = ctx.deps
    df = deps._matched_df
    if df is None or "block_id" not in df.columns:
        return {"error": "No block assignments — call generate_spatial_blocks first."}

    n_folds = deps.n_folds
    unique_blocks = sorted(df["block_id"].unique())
    block_fold = {bid: i % n_folds for i, bid in enumerate(unique_blocks)}

    df = df.copy()
    df["fold"] = df["block_id"].map(block_fold)
    df["split"] = df["fold"].apply(lambda f: "test" if f == 0 else "train")

    fold_sizes = df.groupby("fold").size().to_dict()
    deps._folded_df = df

    return {
        "n_folds": n_folds,
        "fold_sizes": {str(k): int(v) for k, v in sorted(fold_sizes.items())},
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
    }


@transformer_agent.tool
def write_cleaned_shots_to_db(ctx: RunContext[TransformerDeps]) -> dict:
    """
    Write the folded, band-matched DataFrame from assign_folds to the
    gedi_shots_cleaned table. Must be called after assign_folds.
    Returns rows_written.
    """
    from canopy_height_prediction.db import GediShotCleaned, get_session

    deps = ctx.deps
    df = deps._folded_df
    if df is None:
        return {"error": "No folded data — call assign_folds first."}

    rows = []
    for _, row in df.iterrows():
        rows.append(
            GediShotCleaned(
                run_id=deps.run_id,
                shot_id=str(row.get("shot_id", "")),
                lat=row.get("lat"),
                lon=row.get("lon"),
                rh98=row.get("rh98"),
                b2=row.get("B2"),
                b3=row.get("B3"),
                b4=row.get("B4"),
                b8=row.get("B8"),
                b11=row.get("B11"),
                b12=row.get("B12"),
                ndvi=row.get("ndvi"),
                evi=row.get("evi"),
                fold=int(row["fold"]) if row.get("fold") is not None else None,
                split=row.get("split"),
                scl_valid_obs=int(row["valid_obs_count"]) if row.get("valid_obs_count") is not None else None,
            )
        )

    with get_session() as session:
        session.add_all(rows)
        session.commit()

    return {"rows_written": len(rows)}


# ---------------------------------------------------------------------------
# Orchestrator Agent + Tools
# ---------------------------------------------------------------------------

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
