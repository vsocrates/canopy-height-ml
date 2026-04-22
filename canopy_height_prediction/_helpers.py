"""
Pure helper functions shared across agents. No agent, no DB, no Pydantic.
"""

MIN_VALID_OBS = 2  # minimum clear-sky S2 scenes required for a reliable median composite


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
