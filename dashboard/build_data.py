from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib
from typing import Any

import pandas as pd

from alphagraph.kg.report import build_kg_graph_payload


DEFAULT_ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "artifacts" / "alphagraph"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "pipeline_config.toml"


def build_dashboard_payload(
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    config = _read_toml(config_path)
    defaults = dict(config.get("defaults", {}))
    dataset = dataset_name or str(defaults.get("dataset_name", ""))
    dataset_dir = Path(artifact_root) / dataset
    profiles = _profile_rows(config)
    kg_payload = build_kg_graph_payload(
        stock_graph_edges_path=dataset_dir / "stock_graph_edges.parquet",
        kg_stock_edges_llm_path=dataset_dir / "kg_stock_edges_llm.parquet",
        report_llm_relations_path=dataset_dir / "report_llm_relations.parquet",
    )
    backtests = _backtest_payloads(
        dataset_dir / "backtest_runs",
        source_namespace="alphagraph",
        daily_k_path=dataset_dir / "daily_k.parquet",
    )
    if not backtests:
        backtests = _fallback_backtests(Path(artifact_root).parent / "report_hypergraph")
    return {
        "dataset": {
            "name": dataset,
            "path": str(dataset_dir),
            "exists": dataset_dir.exists(),
        },
        "defaults": _json_safe(defaults),
        "profiles": profiles,
        "kg": _compact_kg_payload(kg_payload),
        "clusters": _cluster_payload(dataset_dir / "stock_clusters.parquet"),
        "backtests": backtests,
        "artifacts": _artifact_status(dataset_dir),
    }


def write_dashboard_payload(
    output_path: Path,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    payload = build_dashboard_payload(
        artifact_root=artifact_root,
        config_path=config_path,
        dataset_name=dataset_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static JSON data for alphagraph dashboard")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = write_dashboard_payload(
        output_path=args.output,
        artifact_root=args.artifact_root,
        config_path=args.config_path,
        dataset_name=args.dataset_name,
    )
    print(f"dashboard data written to: {args.output}")
    print(f"dataset: {payload['dataset']['name']}")


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _profile_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = dict(config.get("defaults", {}))
    rows = []
    for name, values in sorted(config.get("profiles", {}).items()):
        merged = dict(defaults)
        merged.update(values)
        rows.append(
            {
                "name": str(name),
                "fusion_type": str(merged.get("fusion_type", "")),
                "graph_module_type": str(merged.get("graph_module_type", "")),
                "edge_type_prior_mode": str(merged.get("edge_type_prior_mode", "")),
                "cluster_method": str(merged.get("cluster_method", "")),
                "trading_strategy": str(merged.get("trading_strategy", "")),
                "execution_mode": str(merged.get("execution_mode", "")),
                "topk": _json_safe(merged.get("topk", "")),
            }
        )
    return rows


def _compact_kg_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": payload.get("summary", {}),
        "months": payload.get("months", []),
        "edge_types": payload.get("edge_types", []),
        "nodes": payload.get("nodes", []),
        "links": payload.get("links", []),
        "edge_rows": payload.get("edge_rows", []),
        "relation_rows": payload.get("relation_rows", [])[:1000],
    }


def _cluster_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": [], "summary": {}}
    frame = _read_frame(path)
    if frame.empty:
        return {"path": str(path), "exists": True, "rows": [], "summary": {"row_count": 0}}
    rows = _frame_records(frame.head(1000))
    summary = {
        "row_count": int(len(frame)),
        "cluster_count": int(frame["cluster_id"].nunique()) if "cluster_id" in frame.columns else 0,
        "method_counts": _value_counts(frame, "cluster_method"),
        "month_counts": _value_counts(frame, "month"),
    }
    return {"path": str(path), "exists": True, "rows": rows, "summary": summary}


def _backtest_payloads(
    run_root: Path,
    source_namespace: str = "alphagraph",
    dataset_prefix: str = "",
    daily_k_path: Path | None = None,
) -> list[dict[str, Any]]:
    if not run_root.exists():
        return []
    payloads = []
    daily_k_df = _read_optional_frame(daily_k_path)
    for run_dir in sorted((path for path in run_root.iterdir() if path.is_dir()), reverse=True):
        metrics_path = run_dir / "backtest_metrics.json"
        returns_path = run_dir / "backtest_returns.parquet"
        if not metrics_path.exists() and not returns_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        returns_df = _read_frame(returns_path) if returns_path.exists() else pd.DataFrame()
        returns = _frame_records(returns_df.head(240)) if not returns_df.empty else []
        html_reports = [str(path) for path in sorted(run_dir.glob("*.html"))]
        payloads.append(
            {
                "run_tag": f"{dataset_prefix}/{run_dir.name}" if dataset_prefix else run_dir.name,
                "source_namespace": source_namespace,
                "path": str(run_dir),
                "metrics": _json_safe(metrics),
                "strategy": _run_strategy_payload(metrics, run_dir.name),
                "returns": returns,
                "kline": _kline_payload(returns_df, daily_k_df),
                "html_reports": html_reports,
            }
        )
    return payloads[:20]


def _fallback_backtests(report_root: Path) -> list[dict[str, Any]]:
    if not report_root.exists():
        return []
    payloads: list[dict[str, Any]] = []
    for dataset_dir in sorted((path for path in report_root.iterdir() if path.is_dir()), reverse=True):
        payloads.extend(
            _backtest_payloads(
                dataset_dir / "backtest_runs",
                source_namespace="report_hypergraph",
                dataset_prefix=dataset_dir.name,
                daily_k_path=dataset_dir / "daily_k.parquet",
            )
        )
    return payloads[:20]


def _artifact_status(dataset_dir: Path) -> dict[str, bool]:
    names = [
        "reports.parquet",
        "stock_graph_edges.parquet",
        "report_llm_relations.parquet",
        "kg_stock_edges_llm.parquet",
        "stock_clusters.parquet",
        "cluster_graph_edges.parquet",
        "monthly_stock_features.parquet",
        "monthly_labels.parquet",
    ]
    return {name: (dataset_dir / name).exists() for name in names}


def _run_strategy_payload(metrics: dict[str, Any], fallback_name: str) -> dict[str, Any]:
    run_config = dict(metrics.get("run_config", {})) if isinstance(metrics.get("run_config"), dict) else {}
    max_hold_months = run_config.get("max_hold_months", "")
    try:
        max_hold_value = int(max_hold_months)
    except Exception:
        max_hold_value = 0
    entry_batch_days = _safe_int(run_config.get("entry_batch_days"), 1)
    stop_loss_confirm_days = _safe_int(run_config.get("stop_loss_confirm_days"), 1)
    cluster_rsj = (
        bool(run_config.get("cluster_require_positive_rsj_20d"))
        or bool(run_config.get("cluster_require_positive_rsj_60d"))
        or bool(run_config.get("cluster_require_positive_ret_20d"))
        or bool(run_config.get("cluster_require_positive_ret_60d"))
    )
    hold_label = "不限持有期" if max_hold_value <= 0 else f"持有 {max_hold_value} 个月"
    variants = []
    if entry_batch_days > 1:
        variants.append(f"分{entry_batch_days}天建仓")
    if stop_loss_confirm_days > 1:
        variants.append(f"{stop_loss_confirm_days}日确认止损")
    if cluster_rsj:
        variants.append("RSJ过滤")
    label = hold_label if not variants else f"{hold_label} / {' / '.join(variants)}"
    key = (
        f"hold_{max_hold_value}m_batch_{entry_batch_days}"
        f"_stop_confirm_{stop_loss_confirm_days}_rsj_{int(cluster_rsj)}"
    )
    return {
        "key": key,
        "label": label,
        "max_hold_months": max_hold_value,
        "stop_loss_pct": run_config.get("stop_loss_pct", ""),
        "stop_loss_confirm_days": stop_loss_confirm_days,
        "entry_batch_days": entry_batch_days,
        "cluster_trend_filter": cluster_rsj,
        "cluster_rsj_filter": cluster_rsj,
        "entry_trigger": run_config.get("entry_trigger", ""),
        "trading_strategy": run_config.get("trading_strategy", ""),
        "config_name": run_config.get("config_name", fallback_name),
    }


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _read_optional_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return _read_frame(path)


def _kline_payload(
    returns_df: pd.DataFrame,
    daily_k_df: pd.DataFrame | None,
    max_codes: int = 24,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"codes": [], "series_by_code": {}}
    if daily_k_df is None or daily_k_df.empty or returns_df.empty:
        return payload
    required_columns = {"date", "code", "open", "high", "low", "close"}
    if not required_columns.issubset(daily_k_df.columns):
        return payload

    events_by_code: dict[str, list[dict[str, str]]] = {}
    ordered_codes: list[str] = []
    for row in returns_df.itertuples(index=False):
        month = str(getattr(row, "month", ""))
        for column, action in (("buy_events", "买入"), ("sell_events", "卖出")):
            for event in _parse_trade_events(getattr(row, column, ""), month, action):
                code = event["code"]
                if code not in events_by_code:
                    events_by_code[code] = []
                    ordered_codes.append(code)
                events_by_code[code].append(event)

    selected_codes = ordered_codes[:max_codes]
    if not selected_codes:
        return payload

    daily = daily_k_df.loc[daily_k_df["code"].astype(str).isin(selected_codes)].copy()
    if daily.empty:
        return payload
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    for column in ("open", "high", "low", "close"):
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    daily = daily.dropna(subset=["date", "code", "open", "high", "low", "close"])
    if daily.empty:
        return payload
    daily["code"] = daily["code"].astype(str)

    series_by_code: dict[str, Any] = {}
    for code in selected_codes:
        code_daily = daily.loc[daily["code"] == code].sort_values("date").copy()
        if code_daily.empty:
            continue
        candles = [
            {
                "time": row.date.strftime("%Y-%m-%d"),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
            }
            for row in code_daily.itertuples(index=False)
        ]
        available_times = [item["time"] for item in candles]
        markers = [
            _trade_marker(event, available_times)
            for event in events_by_code.get(code, [])
            if available_times
        ]
        series_by_code[code] = {"candles": candles, "markers": markers}

    payload["codes"] = list(series_by_code)
    payload["series_by_code"] = series_by_code
    return payload


def _split_codes(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [code.strip() for code in str(value).split(",") if code.strip()]


def _parse_trade_events(value: Any, month: str, action: str) -> list[dict[str, str]]:
    if value is None or pd.isna(value):
        return []
    events: list[dict[str, str]] = []
    for item in str(value).split(";"):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        events.append(
            {
                "month": month,
                "time": parts[0],
                "code": parts[1],
                "action": action,
                "reason": parts[2] if len(parts) > 2 else "",
            }
        )
    return events


def _trade_marker(event: dict[str, str], available_times: list[str]) -> dict[str, str]:
    marker_time = str(event.get("time", ""))
    if marker_time not in set(available_times):
        marker_time = _nearest_available_time(marker_time, available_times)
    is_buy = event.get("action") == "买入"
    reason = event.get("reason", "")
    reason_label = f" {reason}" if reason else ""
    return {
        "time": marker_time,
        "position": "belowBar" if is_buy else "aboveBar",
        "color": "#0f766e" if is_buy else "#be123c",
        "shape": "arrowUp" if is_buy else "arrowDown",
        "text": f"{event.get('month', '')} {event.get('action', '')}{reason_label}".strip(),
    }


def _nearest_available_time(marker_time: str, available_times: list[str]) -> str:
    parsed_marker = pd.to_datetime(marker_time, errors="coerce")
    if pd.isna(parsed_marker):
        return available_times[-1]
    for available_time in available_times:
        parsed_available = pd.to_datetime(available_time, errors="coerce")
        if pd.notna(parsed_available) and parsed_available >= parsed_marker:
            return available_time
    return available_times[-1]


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_safe(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {str(key): int(value) for key, value in frame[column].fillna("").astype(str).value_counts().items()}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":
    main()
