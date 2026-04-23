from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

from canopy_height_prediction._models import QADecision, QADeps

load_dotenv()


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
