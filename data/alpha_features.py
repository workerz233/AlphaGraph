from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphagraph.data.market_data import normalize_baostock_code


EXPECTED_ALPHA_FACTORS: tuple[tuple[str, int], ...] = (
    ("A191.alpha074", -1),
    ("A101.alpha047", 1),
    ("A191.alpha179", -1),
    ("A158.HIGH0", 1),
    ("A101.alpha045", 1),
    ("A191.alpha113", 1),
    ("A101.alpha012", 1),
    ("A158.IMIN10", -1),
    ("A101.alpha022", 1),
    ("A191.alpha104", 1),
)
ALPHA_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    f"alpha_{factor.replace('.', '_')}" for factor, _ in EXPECTED_ALPHA_FACTORS
)
EXPECTED_ALPHA_RANKS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 9)


def load_alpha_factor_manifest(results_dir: str | Path, top_n: int = 10) -> dict[str, Any]:
    if int(top_n) != len(EXPECTED_ALPHA_FACTORS):
        raise ValueError(f"Only the fixed top-{len(EXPECTED_ALPHA_FACTORS)} alpha factors are supported")
    results_path = Path(results_dir)
    ranking_path = results_path / "factor_ranking.csv"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Alpha ranking file not found: {ranking_path}")
    ranking = pd.read_csv(ranking_path)
    required = {"factor", "direction", "validation_rank"}
    missing = required.difference(ranking.columns)
    if missing:
        raise ValueError(f"Alpha ranking missing columns: {sorted(missing)}")
    top = (
        ranking.assign(
            factor=ranking["factor"].astype(str),
            validation_rank=pd.to_numeric(ranking["validation_rank"], errors="coerce"),
            direction=pd.to_numeric(ranking["direction"], errors="coerce"),
        )
        .dropna(subset=["validation_rank", "direction"])
        .sort_values(["validation_rank", "factor"], kind="mergesort")
        .head(len(EXPECTED_ALPHA_FACTORS))
    )
    expected_names = [factor for factor, _ in EXPECTED_ALPHA_FACTORS]
    actual_names = top["factor"].tolist()
    actual_ranks = top["validation_rank"].astype(int).tolist()
    expected_directions = {factor: direction for factor, direction in EXPECTED_ALPHA_FACTORS}
    actual_directions = dict(zip(top["factor"], top["direction"].astype(int), strict=False))
    if (
        actual_names != expected_names
        or actual_directions != expected_directions
        or actual_ranks != list(EXPECTED_ALPHA_RANKS)
    ):
        raise ValueError(
            "Alpha ranking does not match the fixed top-10 manifest: "
            f"expected {expected_names}, got {actual_names}"
        )
    return {
        "factors": expected_names,
        "directions": expected_directions,
        "validation_ranks": list(EXPECTED_ALPHA_RANKS),
        "columns": list(ALPHA_FEATURE_COLUMNS),
        "top_n": len(EXPECTED_ALPHA_FACTORS),
        "source": str(ranking_path),
        "aggregation": "month_end_last_valid",
        "direction_aligned": True,
    }


def build_monthly_alpha_features(
    daily_df: pd.DataFrame,
    monthly_features_df: pd.DataFrame,
    results_dir: str | Path,
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest = load_alpha_factor_manifest(results_dir, top_n=top_n)
    required = {"date", "code", "open", "high", "low", "close", "volume"}
    missing = required.difference(daily_df.columns)
    if missing:
        raise ValueError(f"Daily K data missing columns for alpha factors: {sorted(missing)}")
    if monthly_features_df.empty:
        empty = monthly_features_df.copy()
        for column in ALPHA_FEATURE_COLUMNS:
            empty[column] = pd.Series(dtype=float)
        return empty, _empty_audit(), manifest

    daily = daily_df.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="raise")
    daily["asset"] = daily["code"].map(_asset_code)
    daily = daily.drop_duplicates(["date", "asset"]).sort_values(["date", "asset"])
    raw = _compute_daily_alpha_values(daily, manifest["factors"])
    raw = raw.reindex(columns=manifest["factors"])
    raw = raw.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    for factor, direction in manifest["directions"].items():
        raw[factor] = raw[factor] * int(direction)

    monthly = monthly_features_df.copy()
    monthly["date"] = pd.to_datetime(monthly["date"], errors="raise")
    monthly["code"] = monthly["code"].map(normalize_baostock_code)
    long = raw.rename_axis(index=["date", "asset"]).reset_index()
    long["code"] = long["asset"].map(normalize_baostock_code)
    long = long.drop(columns=["asset"])
    long = long.sort_values(["code", "date"])
    long.loc[:, manifest["factors"]] = long.groupby("code", sort=False)[
        manifest["factors"]
    ].ffill()
    # merge_asof requires the merge key to be globally monotonic, even when a
    # by-key is present.
    long = long.sort_values(["date", "code"])
    monthly_keys = monthly[["month", "date", "code"]].copy()
    monthly_keys["_row_id"] = np.arange(len(monthly_keys), dtype=np.int64)
    monthly_keys = monthly_keys.sort_values(["date", "code"])
    merged = pd.merge_asof(
        monthly_keys,
        long,
        on="date",
        by="code",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.sort_values("_row_id").reset_index(drop=True)
    values = merged[manifest["factors"]].copy()
    values.columns = list(ALPHA_FEATURE_COLUMNS)
    values = values.fillna(0.0)
    result = monthly.reset_index(drop=True).copy()
    result.loc[:, list(ALPHA_FEATURE_COLUMNS)] = values.to_numpy(dtype=float)

    audit_rows: list[dict[str, Any]] = []
    for month, rows in merged.groupby("month", sort=True):
        for factor, column in zip(manifest["factors"], ALPHA_FEATURE_COLUMNS, strict=True):
            valid = rows[factor].notna()
            audit_rows.append(
                {
                    "month": str(month),
                    "factor": factor,
                    "coverage": float(valid.mean()) if len(rows) else 0.0,
                    "valid_rows": int(valid.sum()),
                    "total_rows": int(len(rows)),
                }
            )
    return result, pd.DataFrame(audit_rows), manifest


def _compute_daily_alpha_values(daily: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    from alpha.src.alpha158_daily import compute_alpha158_daily
    from alpha.src.alpha_factor_research import compute_worldquant_daily

    outputs: list[pd.DataFrame] = []
    alpha158_names = {name.split(".", 1)[1] for name in factors if name.startswith("A158.")}
    if alpha158_names:
        values = compute_alpha158_daily(daily).loc[:, sorted(alpha158_names)]
        values.columns = [f"A158.{name}" for name in values.columns]
        outputs.append(values)
    worldquant_names = {name for name in factors if not name.startswith("A158.")}
    if worldquant_names:
        values, audit = compute_worldquant_daily(
            daily,
            libraries=("A101", "A191"),
            factor_names=worldquant_names,
        )
        failures = audit.loc[audit["status"] != "ok"]
        if not failures.empty:
            raise ValueError(f"Alpha factor computation failed: {failures[['factor', 'error']].to_dict('records')}")
        outputs.append(values)
    if not outputs:
        raise ValueError("No alpha factors requested")
    return pd.concat(outputs, axis=1).reindex(columns=factors)


def _asset_code(value: Any) -> str:
    raw = str(value).strip().lower()
    return raw.split(".")[-1].zfill(6)


def _empty_audit() -> pd.DataFrame:
    return pd.DataFrame(columns=["month", "factor", "coverage", "valid_rows", "total_rows"])


def write_alpha_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
