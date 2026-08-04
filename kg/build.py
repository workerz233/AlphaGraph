from __future__ import annotations

import argparse
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from alphagraph.gnn.text_embedder import DEFAULT_EMBEDDING_MODEL_NAME, EmbedderFn, embed_report_texts
from alphagraph.kg.extractor import extract_report_llm_relations
from alphagraph.kg.report import write_kg_html_report
from alphagraph.model.core import build_configured_stock_graph_edges, build_kg_llm_stock_edges


ExtractorFn = Callable[..., pd.DataFrame]
ProgressFn = Callable[[str], None]


def build_kg_artifacts(
    reports_path: str | Path,
    incidence_path: str | Path,
    monthly_features_path: str | Path,
    daily_k_path: str | Path,
    hs300_daily_k_path: str | Path,
    report_embeddings_path: str | Path,
    report_llm_relations_path: str | Path,
    kg_stock_edges_llm_path: str | Path,
    stock_graph_edges_path: str | Path,
    kg_html_path: str | Path,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    stock_edge_builders: str = "report_co_coverage,market_corr_topk,same_industry_topk,kg_llm",
    market_corr_lookback_days: int = 60,
    market_corr_topk: int = 10,
    market_corr_min_score: float = 0.1,
    industry_topk: int = 5,
    llm_kg_enabled: bool = True,
    llm_kg_base_url: str = "",
    llm_kg_api_key: str = "",
    llm_kg_model: str = "",
    llm_kg_max_chars: int = 12000,
    llm_kg_timeout_seconds: int = 120,
    llm_kg_min_confidence: float = 0.0,
    llm_kg_cache_dir: str | Path | None = None,
    llm_kg_raw_log_dir: str | Path | None = None,
    llm_kg_decay_months: int = 3,
    force_extract: bool = False,
    embedding_fn: EmbedderFn | None = None,
    extractor_fn: ExtractorFn = extract_report_llm_relations,
    progress_fn: ProgressFn | None = None,
) -> dict[str, Any]:
    reports_path = Path(reports_path)
    incidence_path = Path(incidence_path)
    monthly_features_path = Path(monthly_features_path)
    daily_k_path = Path(daily_k_path)
    hs300_daily_k_path = Path(hs300_daily_k_path)
    report_embeddings_path = Path(report_embeddings_path)
    report_llm_relations_path = Path(report_llm_relations_path)
    kg_stock_edges_llm_path = Path(kg_stock_edges_llm_path)
    stock_graph_edges_path = Path(stock_graph_edges_path)
    kg_html_path = Path(kg_html_path)

    _progress(progress_fn, "[kg] 读取输入数据...")
    reports_df = _read_frame(reports_path)
    incidence_df = _read_frame(incidence_path)
    monthly_features_df = _read_frame(monthly_features_path)
    daily_k_df = _read_frame(daily_k_path)
    hs300_daily_df = _read_frame(hs300_daily_k_path)
    _progress(
        progress_fn,
        "[kg] 输入数据读取完成："
        f"reports={len(reports_df)}, incidence={len(incidence_df)}, "
        f"monthly_features={len(monthly_features_df)}, daily_k={len(daily_k_df)}, "
        f"hs300_daily_k={len(hs300_daily_df)}",
    )

    cached_embeddings_df = (
        _read_frame(report_embeddings_path)
        if report_embeddings_path.exists()
        else pd.DataFrame()
    )
    report_cache_matches = _embedding_cache_matches(
        reports_df,
        cached_embeddings_df,
        model_name=model_name,
    )
    if report_embeddings_path.exists() and report_cache_matches:
        _progress(progress_fn, f"[kg] 复用报告 embedding：{report_embeddings_path}")
        embeddings_df = cached_embeddings_df
    else:
        if report_embeddings_path.exists():
            _progress(
                progress_fn,
                "[kg] 报告集合已变化，或正文/embedding 配置不匹配，重新生成缓存。",
            )
        _progress(
            progress_fn,
            f"[kg] 开始生成报告 embedding：reports={len(reports_df)}, model={model_name}",
        )
        embeddings_df = embed_report_texts(
            reports_df=reports_df,
            model_name=model_name,
            output_path=report_embeddings_path,
            embedder=embedding_fn,
        )
        embeddings_df = _add_embedding_cache_metadata(
            embeddings_df,
            reports_df,
            model_name=model_name,
        )
        _write_frame(embeddings_df, report_embeddings_path)
        _progress(progress_fn, f"[kg] 报告 embedding 生成完成：rows={len(embeddings_df)}")

    if report_llm_relations_path.exists() and not force_extract and report_cache_matches:
        _progress(progress_fn, f"[kg] 复用 LLM 关系抽取结果：{report_llm_relations_path}")
        llm_relations_df = _read_frame(report_llm_relations_path)
    else:
        _progress(
            progress_fn,
            "[kg] 开始 LLM 关系抽取："
            f"enabled={llm_kg_enabled}, reports={len(reports_df)}, "
            f"model={llm_kg_model}, timeout={llm_kg_timeout_seconds}s",
        )
        llm_relations_df = _call_extractor(
            extractor_fn,
            reports_df=reports_df,
            base_url=llm_kg_base_url,
            api_key=llm_kg_api_key or os.environ.get("LLM_KG_API_KEY", ""),
            model=llm_kg_model,
            max_chars=llm_kg_max_chars,
            timeout=llm_kg_timeout_seconds,
            min_confidence=llm_kg_min_confidence,
            cache_dir=llm_kg_cache_dir,
            raw_log_dir=llm_kg_raw_log_dir,
            enabled=llm_kg_enabled,
            progress_callback=progress_fn,
        )
        _write_frame(llm_relations_df, report_llm_relations_path)
        _progress(progress_fn, f"[kg] LLM 关系抽取完成：relations={len(llm_relations_df)}")

    _progress(progress_fn, "[kg] 开始构建 KG 股票边...")
    kg_edges_df = build_kg_llm_stock_edges(
        monthly_features_df=monthly_features_df,
        llm_relations_df=llm_relations_df,
        llm_kg_decay_months=llm_kg_decay_months,
    )
    _write_frame(kg_edges_df, kg_stock_edges_llm_path)
    _progress(progress_fn, f"[kg] KG 股票边写入完成：rows={len(kg_edges_df)}")

    _progress(progress_fn, f"[kg] 开始构建完整股票图：builders={stock_edge_builders}")
    stock_graph_edges_df = build_configured_stock_graph_edges(
        monthly_features_df=monthly_features_df,
        reports_df=reports_df,
        incidence_df=incidence_df,
        report_embeddings_df=embeddings_df,
        daily_k_df=daily_k_df,
        hs300_daily_df=hs300_daily_df,
        edge_builders=stock_edge_builders,
        market_corr_lookback_days=market_corr_lookback_days,
        market_corr_topk=market_corr_topk,
        market_corr_min_score=market_corr_min_score,
        industry_topk=industry_topk,
        llm_relations_df=llm_relations_df,
        llm_kg_decay_months=llm_kg_decay_months,
    )
    _write_frame(stock_graph_edges_df, stock_graph_edges_path)
    _progress(progress_fn, f"[kg] 完整股票图写入完成：rows={len(stock_graph_edges_df)}")

    _progress(progress_fn, "[kg] 开始生成 KG HTML...")
    write_kg_html_report(
        output_path=kg_html_path,
        stock_graph_edges_path=stock_graph_edges_path,
        kg_stock_edges_llm_path=kg_stock_edges_llm_path,
        report_llm_relations_path=report_llm_relations_path,
    )
    _progress(progress_fn, f"[kg] KG HTML 写入完成：{kg_html_path}")

    return {
        "reports_path": str(reports_path),
        "incidence_path": str(incidence_path),
        "monthly_features_path": str(monthly_features_path),
        "daily_k_path": str(daily_k_path),
        "hs300_daily_k_path": str(hs300_daily_k_path),
        "report_embeddings_path": str(report_embeddings_path),
        "report_llm_relations_path": str(report_llm_relations_path),
        "kg_stock_edges_llm_path": str(kg_stock_edges_llm_path),
        "stock_graph_edges_path": str(stock_graph_edges_path),
        "html_path": str(kg_html_path),
        "report_rows": int(len(reports_df)),
        "embedding_rows": int(len(embeddings_df)),
        "relation_rows": int(len(llm_relations_df)),
        "kg_edge_rows": int(len(kg_edges_df)),
        "stock_graph_edge_rows": int(len(stock_graph_edges_df)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone alphagraph KG artifacts")
    parser.add_argument("--reports-path", type=Path, required=True)
    parser.add_argument("--incidence-path", type=Path, required=True)
    parser.add_argument("--monthly-features-path", type=Path, required=True)
    parser.add_argument("--daily-k-path", type=Path, required=True)
    parser.add_argument("--hs300-daily-k-path", type=Path, required=True)
    parser.add_argument("--report-embeddings-path", type=Path, required=True)
    parser.add_argument("--report-llm-relations-path", type=Path, required=True)
    parser.add_argument("--kg-stock-edges-llm-path", type=Path, required=True)
    parser.add_argument("--stock-graph-edges-path", type=Path, required=True)
    parser.add_argument("--kg-html-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--stock-edge-builders", type=str, default="report_co_coverage,market_corr_topk,same_industry_topk,kg_llm")
    parser.add_argument("--market-corr-lookback-days", type=int, default=60)
    parser.add_argument("--market-corr-topk", type=int, default=10)
    parser.add_argument("--market-corr-min-score", type=float, default=0.1)
    parser.add_argument("--industry-topk", type=int, default=5)
    parser.add_argument("--llm-kg-enabled", type=_parse_bool, default=True)
    parser.add_argument("--llm-kg-base-url", type=str, default="")
    parser.add_argument("--llm-kg-api-key", type=str, default="")
    parser.add_argument("--llm-kg-model", type=str, default="")
    parser.add_argument("--llm-kg-max-chars", type=int, default=12000)
    parser.add_argument("--llm-kg-timeout-seconds", type=int, default=120)
    parser.add_argument("--llm-kg-min-confidence", type=float, default=0.0)
    parser.add_argument("--llm-kg-cache-dir", type=Path, default=None)
    parser.add_argument("--llm-kg-raw-log-dir", type=Path, default=None)
    parser.add_argument("--llm-kg-decay-months", type=int, default=3)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()

    result = build_kg_artifacts(
        reports_path=args.reports_path,
        incidence_path=args.incidence_path,
        monthly_features_path=args.monthly_features_path,
        daily_k_path=args.daily_k_path,
        hs300_daily_k_path=args.hs300_daily_k_path,
        report_embeddings_path=args.report_embeddings_path,
        report_llm_relations_path=args.report_llm_relations_path,
        kg_stock_edges_llm_path=args.kg_stock_edges_llm_path,
        stock_graph_edges_path=args.stock_graph_edges_path,
        kg_html_path=args.kg_html_path,
        model_name=args.model_name,
        stock_edge_builders=args.stock_edge_builders,
        market_corr_lookback_days=args.market_corr_lookback_days,
        market_corr_topk=args.market_corr_topk,
        market_corr_min_score=args.market_corr_min_score,
        industry_topk=args.industry_topk,
        llm_kg_enabled=args.llm_kg_enabled,
        llm_kg_base_url=args.llm_kg_base_url,
        llm_kg_api_key=args.llm_kg_api_key,
        llm_kg_model=args.llm_kg_model,
        llm_kg_max_chars=args.llm_kg_max_chars,
        llm_kg_timeout_seconds=args.llm_kg_timeout_seconds,
        llm_kg_min_confidence=args.llm_kg_min_confidence,
        llm_kg_cache_dir=args.llm_kg_cache_dir,
        llm_kg_raw_log_dir=args.llm_kg_raw_log_dir,
        llm_kg_decay_months=args.llm_kg_decay_months,
        force_extract=args.force_extract,
        progress_fn=_stdout_progress,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _read_frame(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _write_frame(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_pickle(path)


def _add_embedding_cache_metadata(
    embeddings_df: pd.DataFrame,
    reports_df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    enriched = embeddings_df.copy()
    text_hashes = _report_text_hashes(reports_df)
    enriched["embedding_model"] = str(model_name)
    enriched["embedding_source_sha256"] = enriched["report_id"].astype(str).map(text_hashes)
    return enriched


def _embedding_cache_matches(
    reports_df: pd.DataFrame,
    cached_df: pd.DataFrame,
    model_name: str,
) -> bool:
    required_columns = {
        "report_id",
        "embedding_model",
        "embedding_source_sha256",
    }
    if "report_id" not in reports_df.columns or not required_columns.issubset(
        cached_df.columns
    ):
        return False
    current_ids = set(reports_df["report_id"].dropna().astype(str))
    cached_ids = set(cached_df["report_id"].dropna().astype(str))
    if not current_ids or current_ids != cached_ids:
        return False
    if set(cached_df["embedding_model"].dropna().astype(str)) != {str(model_name)}:
        return False
    cached_hashes = (
        cached_df[["report_id", "embedding_source_sha256"]]
        .drop_duplicates(subset=["report_id"], keep="last")
        .set_index("report_id")["embedding_source_sha256"]
        .astype(str)
        .to_dict()
    )
    return cached_hashes == _report_text_hashes(reports_df)


def _report_text_hashes(reports_df: pd.DataFrame) -> dict[str, str]:
    return {
        str(row.report_id): sha256(str(row.raw_text).encode("utf-8")).hexdigest()
        for row in reports_df.reindex(columns=["report_id", "raw_text"]).itertuples(index=False)
    }


def _call_extractor(extractor_fn: ExtractorFn, **kwargs: Any) -> pd.DataFrame:
    try:
        signature = inspect.signature(extractor_fn)
    except (TypeError, ValueError):
        return extractor_fn(**kwargs)
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return extractor_fn(**kwargs)
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return extractor_fn(**filtered_kwargs)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _progress(progress_fn: ProgressFn | None, message: str) -> None:
    if progress_fn is not None:
        progress_fn(message)


def _stdout_progress(message: str) -> None:
    print(message, flush=True)


if __name__ == "__main__":
    main()
