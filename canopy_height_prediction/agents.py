from dataclasses import dataclass, field
import os
from typing import Literal, Optional

import logfire
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv()

if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure()
    logfire.instrument_pydantic_ai()

# ---------------------------------------------------------------------------
# GEE Tool Return Models
# ---------------------------------------------------------------------------


class GediQueryResult(BaseModel):
    """Return type of query_gedi_earthengine tool."""

    total_shots: int
    aoi_area_km2: float
    raw_shot_density_per_km2: float
    # Each bin is [bin_center, count] — 10 bins per histogram
    rh98_histogram: list[list]
    sensitivity_histogram: list[list]
    slope_histogram: list[list]
    quality_flag_counts: dict[str, int]


class Sentinel2QueryResult(BaseModel):
    """Return type of query_sentinel2 tool."""

    scene_count: int
    mean_cloud_cover_pct: float
    temporal_coverage_days: Optional[int]
    bands_available: list[str]


# ---------------------------------------------------------------------------
# Ingestor — Deps + Output
# ---------------------------------------------------------------------------


@dataclass
class IngestorDeps:
    run_id: str
    aoi_bbox: tuple                          # (min_lon, min_lat, max_lon, max_lat)
    date_start: str
    date_end: str
    replan_count: int = 0                    # agent is more lenient when > 0
    sensitivity_min_suggested: float = 0.95
    slope_max_deg_suggested: float = 30.0
    min_shot_density_per_km2: float = 1.0
    gee_project: str = "canopy-height-ml"
    # Populated by apply_quality_filters tool so write_raw_shots_to_db can access it
    _filtered_shots: Optional[object] = field(default=None, repr=False)


class IngestorDecision(BaseModel):
    passed: bool
    sensitivity_min: float
    slope_max_deg: float
    raw_shots: int
    accepted_shots: int
    rationale: str
    recommended_action: Literal[
        "proceed",
        "replan_widen_date",
        "replan_relax_thresholds",
        "abort",
    ]
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Transformer — Deps + Output (stub)
# ---------------------------------------------------------------------------


@dataclass
class TransformerDeps:
    run_id: str
    aoi_bbox: tuple
    date_start: str
    date_end: str
    replan_count: int = 0
    n_folds: int = 5
    gee_project: str = "canopy-height-ml"
    # Populated by extract_sentinel_bands; consumed by compute_variogram / assign_folds
    _matched_df: Optional[object] = field(default=None, repr=False)
    # Populated by assign_folds; consumed by write_cleaned_shots_to_db
    _folded_df: Optional[object] = field(default=None, repr=False)


class TransformerDecision(BaseModel):
    passed: bool
    cv_block_size_km: float
    n_folds: int
    rationale: str
    recommended_action: Literal[
        "proceed",
        "replan_widen_date",
        "replan_relax_thresholds",
        "abort",
    ]
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# QA — Deps + Output (stub)
# ---------------------------------------------------------------------------


@dataclass
class QADeps:
    run_id: str
    replan_count: int = 0
    r2_threshold: float = 0.6
    rmse_threshold_m: float = 5.0


class QADecision(BaseModel):
    passed: bool
    issues: list[str] = Field(default_factory=list)
    rationale: str
    recommended_action: Literal[
        "proceed",
        "replan_widen_date",
        "replan_relax_thresholds",
        "abort",
    ]


# ---------------------------------------------------------------------------
# Orchestrator — Deps + Output (stub)
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorDeps:
    run_id: str
    aoi_bbox: tuple
    date_range: tuple
    max_replans: int = 3


class OrchestratorDecision(BaseModel):
    action: Literal["proceed", "replan", "abort"]
    rationale: str
    updated_date_start: Optional[str] = None
    updated_date_end: Optional[str] = None
    updated_sensitivity_min: Optional[float] = None
    updated_slope_max_deg: Optional[float] = None


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

def _fc_to_gdf(fc):
    """Download a GEE FeatureCollection to a GeoDataFrame, paginating in 5000-element chunks."""
    import geopandas as gpd

    total = fc.size().getInfo()
    features = []
    chunk = 5000
    for offset in range(0, total, chunk):
        page = fc.toList(min(chunk, total - offset), offset).getInfo()
        features.extend(page)
    return gpd.GeoDataFrame.from_features(features)


def _load_gedi_shots(aoi, date_start: str, date_end: str):
    """
    Load GEDI L2A vector shots via the index collection.
    LARSE/GEDI/GEDI02_A_002 is an IndexedFolder — it must be accessed by
    querying the index first, then merging the matching sub-collections.
    """
    import ee

    # filterDate() doesn't work on this index — features store dates in a
    # 'time_start' property, not system:time_start. Use property filters instead.
    index = (
        ee.FeatureCollection("LARSE/GEDI/GEDI02_A_002_INDEX")
        .filterBounds(aoi)
        .filter(ee.Filter.gte("time_start", date_start))
        .filter(ee.Filter.lt("time_start", date_end))
    )
    table_ids = index.aggregate_array("table_id").getInfo()
    if not table_ids:
        return ee.FeatureCollection([])
    tables = [ee.FeatureCollection(tid).filterBounds(aoi) for tid in table_ids]
    return ee.FeatureCollection(tables).flatten()


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
# Transformer helpers
# ---------------------------------------------------------------------------


def _estimate_variogram_range(lats, lons, values, n_lags: int = 15, subsample: int = 2000):
    """
    Compute a binned experimental variogram and return the estimated range in km.
    Uses flat-earth approximation (fine for AOIs < ~500km across).
    Returns (range_km, bin_centers, bin_semivariances).
    """
    import numpy as np

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    values = np.asarray(values, dtype=float)

    n = len(lats)
    if n > subsample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, subsample, replace=False)
        lats, lons, values = lats[idx], lons[idx], values[idx]
        n = subsample

    mean_lat_rad = np.deg2rad(np.mean(lats))
    dx = (lons[:, None] - lons[None, :]) * 111.0 * np.cos(mean_lat_rad)
    dy = (lats[:, None] - lats[None, :]) * 111.0
    dist_km = np.sqrt(dx**2 + dy**2)

    diff = values[:, None] - values[None, :]
    sv = 0.5 * diff**2

    triu = np.triu_indices(n, k=1)
    dist_flat = dist_km[triu]
    sv_flat = sv[triu]

    max_lag = float(np.percentile(dist_flat, 50))
    bins = np.linspace(0, max_lag, n_lags + 1)
    bin_centers, bin_sv = [], []
    for k in range(n_lags):
        mask = (dist_flat >= bins[k]) & (dist_flat < bins[k + 1])
        if mask.sum() >= 5:
            bin_centers.append(round(float((bins[k] + bins[k + 1]) / 2), 3))
            bin_sv.append(round(float(np.mean(sv_flat[mask])), 4))

    if not bin_sv:
        return 5.0, [], []

    sill = max(bin_sv)
    range_km = bin_centers[-1]
    for center, sv_val in zip(bin_centers, bin_sv):
        if sv_val >= 0.95 * sill:
            range_km = center
            break

    return round(range_km, 2), bin_centers, bin_sv


# ---------------------------------------------------------------------------
# Transformer Agent + Tools
# ---------------------------------------------------------------------------

MIN_VALID_OBS = 2  # minimum clear-sky S2 scenes required for a reliable median composite

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
    import numpy as np
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
    import numpy as np

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
