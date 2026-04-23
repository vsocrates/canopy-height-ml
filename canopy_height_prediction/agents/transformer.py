from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from canopy_height_prediction._helpers import MIN_VALID_OBS, _estimate_variogram_range, _fc_to_gdf

load_dotenv()
from canopy_height_prediction._models import TransformerDecision, TransformerDeps

# Maximum GEDI shots sampled against S2 — XGBoost doesn't benefit beyond this.
# Shots are randomly subsampled before GEE extraction when the ingestor returns more.
_MAX_EXTRACTION_SHOTS = 5000
# Shots per sampleRegions request — keeps GEE payload under 10 MB.
_SAMPLE_BATCH = 500
# Concurrent GEE batch requests (GEE REST API supports parallel interactive calls).
_SAMPLE_WORKERS = 8


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
   - If the result contains an "error" key, set passed=False,
     recommended_action="replan_widen_date", and do not call any other tools.
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

    try:
        with get_session() as session:
            rows = session.query(GediShotRaw).filter_by(run_id=deps.run_id).all()

        if not rows:
            return {"error": f"No raw shots found for run_id={deps.run_id}"}

        n_shots_total = len(rows)

        if n_shots_total > _MAX_EXTRACTION_SHOTS:
            import random
            rows = random.sample(rows, _MAX_EXTRACTION_SHOTS)
            print(f"  Subsampled {n_shots_total} → {_MAX_EXTRACTION_SHOTS} shots before GEE extraction", flush=True)

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
            "2.5 * (B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 10000)",
            {"B8": s2.select("B8"), "B4": s2.select("B4"), "B2": s2.select("B2")},
        ).rename("evi")
        image = s2.addBands([ndvi, evi, valid_obs_count])

        # sampleRegions serializes the entire fc into the GEE request; batching keeps
        # each request under GEE's 10MB payload limit. Batches run in parallel via
        # ThreadPoolExecutor — GEE's REST API supports concurrent interactive requests.
        from concurrent.futures import ThreadPoolExecutor
        import geopandas as gpd

        batches = [features[i:i + _SAMPLE_BATCH] for i in range(0, len(features), _SAMPLE_BATCH)]
        n_batches = len(batches)

        def _run_batch(args: tuple) -> "gpd.GeoDataFrame":
            batch_num, batch = args
            print(f"  sampleRegions batch {batch_num}/{n_batches} ({len(batch)} shots)", flush=True)
            batch_fc = ee.FeatureCollection(batch)
            sampled = image.sampleRegions(collection=batch_fc, scale=20, geometries=True, tileScale=4)
            return _fc_to_gdf(sampled)

        with ThreadPoolExecutor(max_workers=min(_SAMPLE_WORKERS, n_batches)) as pool:
            gdfs = list(pool.map(_run_batch, enumerate(batches, start=1)))

        gdf = pd.concat(gdfs, ignore_index=True) if gdfs else gpd.GeoDataFrame()

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
            "n_shots_total_accepted": n_shots_total,
            "n_shots_sampled": n_shots,
            "n_matched": n_matched,
            "nan_rate_pct": nan_rate,
            "valid_obs_below_threshold_pct": below_pct,
            "min_valid_obs_used": MIN_VALID_OBS,
            "band_preview": preview,
        }
    except Exception as e:
        return {"error": f"GEE extraction failed: {e}"}


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
    import math
    import numpy as np

    deps = ctx.deps
    df = deps._matched_df
    if df is None:
        return {"error": "No matched shots — call extract_sentinel_bands first."}

    min_lon, min_lat, max_lon, max_lat = deps.aoi_bbox
    mean_lat_rad = float(np.deg2rad((min_lat + max_lat) / 2))

    aoi_width_km = (max_lon - min_lon) * 111.0 * float(np.cos(mean_lat_rad))
    aoi_height_km = (max_lat - min_lat) * 111.0
    min_blocks_per_side = math.ceil((deps.n_folds + 1) ** 0.5)
    max_allowed_km = min(aoi_width_km, aoi_height_km) / min_blocks_per_side
    requested_block_size_km = block_size_km
    if block_size_km > max_allowed_km:
        block_size_km = max_allowed_km

    deg_lat_per_km = 1.0 / 111.0
    deg_lon_per_km = 1.0 / (111.0 * float(np.cos(mean_lat_rad)))

    block_deg_lat = block_size_km * deg_lat_per_km
    block_deg_lon = block_size_km * deg_lon_per_km

    block_x = ((df["lon"] - min_lon) / block_deg_lon).astype(int)
    block_y = ((df["lat"] - min_lat) / block_deg_lat).astype(int)
    n_blocks_x = max(int((max_lon - min_lon) / block_deg_lon) + 1, 1)
    df["block_id"] = block_y * n_blocks_x + block_x

    block_counts = df["block_id"].value_counts()
    result = {
        "block_size_km": block_size_km,
        "n_blocks_occupied": int(block_counts.shape[0]),
        "shots_per_block_mean": round(float(block_counts.mean()), 1),
        "shots_per_block_min": int(block_counts.min()),
        "shots_per_block_max": int(block_counts.max()),
    }
    if block_size_km != requested_block_size_km:
        result["block_size_km_capped"] = True
        result["block_size_km_requested"] = requested_block_size_km
        result["note"] = (
            f"block_size_km capped from {requested_block_size_km:.1f} to {block_size_km:.1f} km "
            f"to ensure ≥{min_blocks_per_side}×{min_blocks_per_side} blocks for {deps.n_folds} folds."
        )
    return result


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
