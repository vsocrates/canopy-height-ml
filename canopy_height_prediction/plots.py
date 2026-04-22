"""
Diagnostic figures for visual inspection of pipeline training data.

Call generate_pipeline_figures(run_id, output_dir) after a successful pipeline run.
Writes 4 PNGs:
  shot_map.png      — raw vs. cleaned shots on satellite basemap, colored by rh98
  fold_map.png      — cleaned shots colored by spatial CV fold
  variogram.png     — empirical semivariogram with estimated range + block size
  distributions.png — rh98 / ndvi / evi violin plots split by train vs. test
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sqlalchemy.orm import Session

from canopy_height_prediction.agents import _estimate_variogram_range
from canopy_height_prediction.config import FIGURES_DIR
from canopy_height_prediction.db import GediShotCleaned, GediShotRaw, SessionLocal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_pipeline_figures(
    run_id: str,
    output_dir: Path | None = None,
) -> list[Path]:
    """Load run data from DB and write 4 diagnostic PNGs to output_dir."""
    out = Path(output_dir) if output_dir else FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)

    raw_gdf, cleaned_gdf = _load_shots(run_id)

    paths = [
        _plot_shot_map(raw_gdf, cleaned_gdf, out),
        _plot_fold_map(cleaned_gdf, out),
        _plot_variogram(cleaned_gdf, out),
        _plot_distributions(cleaned_gdf, out),
    ]
    return paths


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_shots(run_id: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    with SessionLocal() as session:
        raw_rows = session.query(GediShotRaw).filter(GediShotRaw.run_id == run_id).all()
        cleaned_rows = (
            session.query(GediShotCleaned)
            .filter(GediShotCleaned.run_id == run_id)
            .all()
        )

    def _to_gdf(rows, cols: list[str]) -> gpd.GeoDataFrame:
        data = [{c: getattr(r, c) for c in cols} for r in rows]
        df = pd.DataFrame(data)
        if df.empty:
            return gpd.GeoDataFrame(df, geometry=[], crs="EPSG:4326")
        geom = [Point(row["lon"], row["lat"]) for _, row in df.iterrows()]
        return gpd.GeoDataFrame(df, geometry=geom, crs="EPSG:4326")

    raw_gdf = _to_gdf(raw_rows, ["lat", "lon", "rh98"])
    cleaned_gdf = _to_gdf(
        cleaned_rows,
        ["lat", "lon", "rh98", "ndvi", "evi", "fold", "split"],
    )
    return raw_gdf, cleaned_gdf


def _try_add_basemap(ax, crs) -> None:
    """Add satellite basemap; silently skip if contextily is unavailable."""
    try:
        import contextily as ctx
        ctx.add_basemap(ax, crs=crs, source=ctx.providers.Esri.WorldImagery, zoom="auto")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Figure 1: shot_map — raw vs. cleaned shots on basemap
# ---------------------------------------------------------------------------


def _plot_shot_map(
    raw_gdf: gpd.GeoDataFrame,
    cleaned_gdf: gpd.GeoDataFrame,
    out: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))

    raw_wm = raw_gdf.to_crs("EPSG:3857") if not raw_gdf.empty else raw_gdf
    cleaned_wm = cleaned_gdf.to_crs("EPSG:3857") if not cleaned_gdf.empty else cleaned_gdf

    _try_add_basemap(ax, crs="EPSG:3857")

    if not raw_wm.empty:
        raw_wm.plot(ax=ax, color="lightgrey", markersize=4, alpha=0.5, label="raw shots")

    if not cleaned_wm.empty:
        cleaned_wm.plot(
            ax=ax,
            column="rh98",
            cmap="YlGn",
            markersize=12,
            alpha=0.85,
            legend=True,
            legend_kwds={"label": "rh98 (m)", "shrink": 0.6},
        )

    ax.set_title("GEDI Shots — Raw (grey) vs. Cleaned (rh98)", fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()

    path = out / "shot_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: fold_map — cleaned shots colored by CV fold
# ---------------------------------------------------------------------------


def _plot_fold_map(cleaned_gdf: gpd.GeoDataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 8))

    if cleaned_gdf.empty:
        ax.set_title("Fold Map (no data)")
        path = out / "fold_map.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    cleaned_wm = cleaned_gdf.to_crs("EPSG:3857")
    _try_add_basemap(ax, crs="EPSG:3857")

    n_folds = int(cleaned_wm["fold"].max()) + 1
    cmap = plt.get_cmap("tab10", n_folds)
    for fold_id in range(n_folds):
        subset = cleaned_wm[cleaned_wm["fold"] == fold_id]
        label = "test" if fold_id == 0 else f"fold {fold_id}"
        subset.plot(ax=ax, color=cmap(fold_id), markersize=10, alpha=0.85, label=label)

    ax.legend(title="Fold", loc="lower right", fontsize=9)
    ax.set_title("Spatial CV Fold Assignment (fold 0 = test)", fontsize=13)
    ax.set_axis_off()
    plt.tight_layout()

    path = out / "fold_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: variogram — empirical semivariogram
# ---------------------------------------------------------------------------


def _plot_variogram(cleaned_gdf: gpd.GeoDataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    path = out / "variogram.png"

    if cleaned_gdf.empty or len(cleaned_gdf) < 10:
        ax.set_title("Variogram (insufficient data)")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    range_km, centers, sv = _estimate_variogram_range(
        cleaned_gdf["lat"].values,
        cleaned_gdf["lon"].values,
        cleaned_gdf["rh98"].values,
    )

    if centers:
        ax.plot(centers, sv, "o-", color="steelblue", linewidth=1.8, markersize=5, label="semivariance")
        ax.axvline(range_km, color="tomato", linestyle="--", linewidth=1.5,
                   label=f"estimated range = {range_km:.1f} km")
        ax.set_xlabel("Lag distance (km)")
        ax.set_ylabel("Semivariance")
    else:
        ax.text(0.5, 0.5, "Too few pairs to plot variogram",
                ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Empirical Semivariogram (rh98)", fontsize=13)
    ax.legend(fontsize=9)
    plt.tight_layout()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: distributions — rh98 / ndvi / evi by train vs. test
# ---------------------------------------------------------------------------


def _plot_distributions(cleaned_gdf: gpd.GeoDataFrame, out: Path) -> Path:
    features = ["rh98", "ndvi", "evi"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    path = out / "distributions.png"

    if cleaned_gdf.empty:
        fig.suptitle("Distributions (no data)")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    df = cleaned_gdf[features + ["split"]].copy()
    splits = sorted(df["split"].dropna().unique())
    palette = {"train": "steelblue", "test": "tomato"}

    for ax, feat in zip(axes, features):
        for split in splits:
            vals = df[df["split"] == split][feat].dropna().values
            if len(vals) == 0:
                continue
            color = palette.get(split, "grey")
            parts = ax.violinplot([vals], positions=[splits.index(split)],
                                  showmedians=True, widths=0.6)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            # strip overlay for small datasets
            if len(vals) <= 200:
                jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(vals))
                ax.scatter(splits.index(split) + jitter, vals,
                           color=color, alpha=0.4, s=8, zorder=3)

        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels(splits)
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel(feat)

    fig.suptitle("Feature Distributions by Split", fontsize=13, y=1.01)
    plt.tight_layout()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
