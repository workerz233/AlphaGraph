from __future__ import annotations

import argparse
import json
import os
import socket
from contextlib import contextmanager
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterator

import pandas as pd

from alphagraph.data.market_data import (
    INDEX_CODE_HS300,
    INDEX_CODE_ZZ500,
    FORWARD_RETURN_HORIZONS,
    bs as baostock_client,
    build_monthly_index_returns,
    build_future_labels,
    build_monthly_stock_features,
    fetch_daily_k_for_codes,
    load_or_fetch_index_benchmark,
    normalize_baostock_code,
)
from alphagraph.data.alpha_features import (
    build_monthly_alpha_features,
    write_alpha_manifest,
)
from alphagraph.data.stock_metadata import (
    build_monthly_stock_metadata_features,
    fetch_index_membership_for_dates,
    fetch_stock_basic_for_codes,
    fetch_stock_industry_for_dates,
)
from alphagraph.data.financial_features import (
    FINANCIAL_QUERY_METHODS,
    build_monthly_financial_features,
    fetch_quarterly_financial_tables,
)
from alphagraph.data.reports import (
    build_hyperedge_tables,
    parse_report_file,
    scan_report_markdown,
)
from alphagraph.data.tushare_fallback import build_tushare_fetchers
from alphagraph.data.extend_daily_k import build_akshare_fetchers


ARTIFACT_BASE_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "alphagraph"
)
REPORT_MONTH_RE = re.compile(r"(?P<year>\d{4})\D*(?P<month>\d{1,2})")
BAOSTOCK_REQUEST_TIMEOUT_SECONDS = 5.0


def build_target_artifact_dir(
    workspace_root: str | Path,
    markdown_dir: str | Path,
    start_month: str | None = None,
    end_month: str | None = None,
) -> Path:
    dataset_name = _dataset_name_from_markdown_dir(markdown_dir, start_month, end_month)
    base_dir = ARTIFACT_BASE_DIR
    if Path(workspace_root) != Path(__file__).resolve().parents[2]:
        base_dir = Path(workspace_root) / "artifacts" / "alphagraph"
    return base_dir / dataset_name


def derive_fetch_date_range(
    report_months: list[str],
    as_of_date: str | None = None,
) -> tuple[str, str]:
    if not report_months:
        raise ValueError("report_months must not be empty")

    normalized_months = [
        _normalize_report_month(month)
        for month in report_months
    ]
    valid_months = [month for month in normalized_months if month]
    if not valid_months:
        raise ValueError(f"No valid report_month values found: {report_months[:5]}")

    periods = sorted(pd.Period(month, freq="M") for month in valid_months)
    start_date = (periods[0].start_time.normalize() - pd.Timedelta(days=90)).date()
    if as_of_date is None:
        end_date = periods[-1].asfreq("M").end_time.normalize().date()
        end_date = (
            pd.Timestamp(end_date) + pd.offsets.MonthEnd(max(FORWARD_RETURN_HORIZONS))
        ).date()
    else:
        resolved_as_of_date = pd.to_datetime(as_of_date, errors="coerce")
        if pd.isna(resolved_as_of_date):
            raise ValueError(f"Invalid as_of_date: {as_of_date}")
        if resolved_as_of_date.to_period("M") != periods[-1]:
            raise ValueError(
                "as_of_date must fall within the last target month: "
                f"{periods[-1]}"
            )
        end_date = resolved_as_of_date.normalize().date()
    return start_date.isoformat(), end_date.isoformat()


def derive_target_months(data_start_month: str, signal_month: str) -> set[str]:
    resolved_start = _normalize_report_month(data_start_month)
    resolved_end = _normalize_report_month(signal_month)
    if not resolved_start or not resolved_end:
        raise ValueError(
            "data_start_month and signal_month must be valid YYYYMM or YYYY-MM values"
        )
    start_period = pd.Period(resolved_start, freq="M")
    end_period = pd.Period(resolved_end, freq="M")
    if start_period > end_period:
        raise ValueError("data_start_month must not be after signal_month")
    return {str(month) for month in pd.period_range(start_period, end_period, freq="M")}


def build_monthly_benchmark_returns(
    labels_df: pd.DataFrame,
    target_months: set[str],
) -> pd.DataFrame:
    benchmark_columns = [
        f"future_benchmark_return_{horizon}m"
        for horizon in FORWARD_RETURN_HORIZONS
        if f"future_benchmark_return_{horizon}m" in labels_df.columns
    ]
    filtered = labels_df.loc[
        labels_df["month"].isin(sorted(target_months)),
        ["month", *benchmark_columns],
    ].drop_duplicates(subset=["month"])
    filtered = filtered.rename(
        columns={
            f"future_benchmark_return_{horizon}m": f"benchmark_return_{horizon}m"
            for horizon in FORWARD_RETURN_HORIZONS
        }
    )
    return filtered.sort_values("month", ignore_index=True)


def build_monthly_dual_benchmark_returns(
    hs300_daily_df: pd.DataFrame,
    zz500_daily_df: pd.DataFrame,
    target_months: set[str],
    required_horizons: tuple[int, ...] = FORWARD_RETURN_HORIZONS,
) -> pd.DataFrame:
    monthly = pd.DataFrame({"month": sorted(target_months)})
    for horizon in FORWARD_RETURN_HORIZONS:
        hs300_returns_df = build_monthly_index_returns(
            hs300_daily_df,
            target_months=target_months,
            return_column=f"hs300_return_{horizon}m",
            horizon_months=horizon,
        )
        zz500_returns_df = build_monthly_index_returns(
            zz500_daily_df,
            target_months=target_months,
            return_column=f"zz500_return_{horizon}m",
            horizon_months=horizon,
        )
        monthly = monthly.merge(hs300_returns_df, on="month", how="left")
        monthly = monthly.merge(zz500_returns_df, on="month", how="left")
    monthly["benchmark_return_1m"] = monthly["hs300_return_1m"]
    for horizon in FORWARD_RETURN_HORIZONS:
        monthly[f"benchmark_return_{horizon}m"] = monthly[f"hs300_return_{horizon}m"]
    required_columns = [
        f"{index_name}_return_{horizon}m"
        for horizon in required_horizons
        for index_name in ("hs300", "zz500")
    ]
    return monthly.dropna(subset=required_columns).reset_index(drop=True)


def load_reports_dataframe(
    workspace_root: str | Path,
    markdown_dir: str | Path,
    start_month: str | None = None,
    end_month: str | None = None,
) -> pd.DataFrame:
    markdown_paths = scan_report_markdown(
        workspace_root,
        markdown_dir,
        start_month=start_month,
        end_month=end_month,
    )
    if not markdown_paths:
        raise ValueError(f"No markdown files found in {Path(markdown_dir)}")
    return pd.DataFrame([parse_report_file(path) for path in markdown_paths])


def prepare_monthly_dataset(
    workspace_root: str | Path,
    markdown_dir: str | Path,
    artifact_dir: str | Path | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
    fetch_daily_k_fn: Callable[[list[str], str, str], pd.DataFrame] | None = None,
    fetch_benchmark_fn: Callable[..., pd.DataFrame] | None = None,
    parse_reports_fn: Callable[[str | Path, str | Path], pd.DataFrame] | None = None,
    fetch_stock_basic_fn: Callable[[list[str]], pd.DataFrame] | None = None,
    fetch_stock_industry_fn: Callable[[list[str]], pd.DataFrame] | None = None,
    fetch_index_membership_fn: Callable[[list[str]], pd.DataFrame] | None = None,
    fetch_financial_tables_fn: Callable[
        [list[str], list[tuple[int, int]]], dict[str, pd.DataFrame]
    ]
    | None = None,
    stock_universe: str = "hs300_zz500",
    market_data_source: str = "akshare",
    tushare_token_url: str | None = None,
    data_start_month: str | None = None,
    signal_month: str | None = None,
    as_of_date: str | None = None,
    prediction_mode: bool = False,
    alpha_factor_results_dir: str | Path | None = None,
    alpha_factor_top_n: int = 10,
) -> dict[str, Path]:
    root_path = Path(workspace_root)
    markdown_path = Path(markdown_dir)
    target_artifact_dir = Path(artifact_dir) if artifact_dir is not None else build_target_artifact_dir(
        root_path,
        markdown_path,
        start_month=start_month,
        end_month=end_month,
    )
    target_artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = _build_artifact_paths(target_artifact_dir)

    parse_reports = parse_reports_fn or load_reports_dataframe
    raw_reports_df = _call_parse_reports(
        parse_reports,
        root_path,
        markdown_path,
        start_month=start_month,
        end_month=end_month,
    )
    if prediction_mode:
        if not data_start_month or not signal_month or not as_of_date:
            raise ValueError(
                "prediction_mode requires data_start_month, signal_month, and as_of_date"
            )
        raw_reports_df = clip_reports_for_prediction(
            raw_reports_df,
            data_start_month=data_start_month,
            signal_month=signal_month,
            as_of_date=as_of_date,
        )
    if raw_reports_df.empty:
        raise ValueError(f"No reports parsed from {markdown_path}")
    reports_df, incidence_df = build_hyperedge_tables(raw_reports_df)
    if "report_month" in reports_df.columns:
        reports_df["report_month"] = reports_df["report_month"].map(_normalize_report_month)
    directory_month = _month_from_dataset_name(markdown_path.name)
    resolved_start_month = _normalize_report_month(start_month)
    resolved_end_month = _normalize_report_month(end_month)
    if resolved_start_month and resolved_end_month:
        reports_df = reports_df.loc[
            reports_df["report_month"].between(resolved_start_month, resolved_end_month)
        ].reset_index(drop=True)
        if reports_df.empty:
            raise ValueError(
                f"No reports for month range {resolved_start_month}~{resolved_end_month} found in {markdown_path}"
            )
        if not incidence_df.empty and "report_id" in incidence_df.columns:
            incidence_df = incidence_df.loc[
                incidence_df["report_id"].isin(reports_df["report_id"])
            ].reset_index(drop=True)
    if directory_month:
        reports_df = reports_df.loc[reports_df["report_month"] == directory_month].reset_index(drop=True)
        if reports_df.empty:
            raise ValueError(
                f"No reports for directory month {directory_month} found in {markdown_path}"
            )
        if not incidence_df.empty and "report_id" in incidence_df.columns:
            incidence_df = incidence_df.loc[
                incidence_df["report_id"].isin(reports_df["report_id"])
            ].reset_index(drop=True)
    if incidence_df.empty:
        raise ValueError(f"No related stock codes found in {markdown_path}")

    if prediction_mode:
        assert data_start_month is not None
        assert signal_month is not None
        target_months = derive_target_months(data_start_month, signal_month)
    else:
        target_months = {
            _normalize_report_month(month)
            for month in reports_df["report_month"].dropna().tolist()
            if _normalize_report_month(month)
        }
    if not target_months:
        raise ValueError(f"No report_month values found in {markdown_path}")

    stock_universe_name = _normalize_stock_universe(stock_universe)
    report_stock_codes = sorted(incidence_df["stock_code"].dropna().astype(str).unique().tolist())
    if not report_stock_codes:
        raise ValueError(f"No stock codes extracted from {markdown_path}")
    stock_codes = list(report_stock_codes)

    start_date, end_date = derive_fetch_date_range(
        sorted(target_months),
        as_of_date=as_of_date if prediction_mode else None,
    )
    normalized_market_data_source = str(market_data_source).strip().lower()
    use_tushare_primary = normalized_market_data_source == "tushare"
    use_akshare_primary = normalized_market_data_source == "akshare"
    tushare_fetchers = _build_tushare_fetchers(tushare_token_url) if use_tushare_primary else None
    akshare_fetchers = build_akshare_fetchers() if use_akshare_primary else None
    primary_fetchers = akshare_fetchers or tushare_fetchers
    target_month_end_dates = _derive_target_month_end_dates(
        sorted(target_months),
        as_of_date=as_of_date if prediction_mode else None,
    )
    daily_fetch = fetch_daily_k_fn or (
        primary_fetchers.fetch_daily_k
        if primary_fetchers is not None
        else fetch_daily_k_for_codes
    )
    benchmark_fetch = fetch_benchmark_fn or (
        primary_fetchers.fetch_benchmark
        if primary_fetchers is not None
        else load_or_fetch_index_benchmark
    )
    stock_basic_fetch = fetch_stock_basic_fn
    stock_industry_fetch = fetch_stock_industry_fn
    index_membership_fetch = fetch_index_membership_fn
    financial_fetch = fetch_financial_tables_fn

    daily_requires_baostock = (
        fetch_daily_k_fn is None
        and tushare_fetchers is None
        and akshare_fetchers is None
        and not paths["daily_k_path"].exists()
    )
    benchmark_requires_baostock = fetch_benchmark_fn is None and (
        primary_fetchers is None
        and (
            not paths["hs300_daily_k_path"].exists()
            or not paths["zz500_daily_k_path"].exists()
        )
    )
    stock_basic_requires_baostock = (
        fetch_stock_basic_fn is None
        and tushare_fetchers is None
        and akshare_fetchers is None
        and not paths["stock_basic_path"].exists()
    )
    stock_industry_requires_baostock = (
        fetch_stock_industry_fn is None
        and tushare_fetchers is None
        and akshare_fetchers is None
        and not _dated_cache_path_covers_months(
            paths["stock_industry_path"],
            target_month_end_dates,
        )
    )
    index_membership_requires_baostock = (
        fetch_index_membership_fn is None
        and tushare_fetchers is None
        and akshare_fetchers is None
        and not _dated_cache_path_covers_months(
            paths["index_membership_path"],
            target_month_end_dates,
        )
    )
    financial_requires_baostock = (
        fetch_financial_tables_fn is None
        and tushare_fetchers is None
        and akshare_fetchers is None
        and not _all_financial_artifacts_exist(paths)
    )

    if stock_basic_fetch is None:
        stock_basic_fetch = (
            primary_fetchers.fetch_stock_basic
            if primary_fetchers is not None
            else lambda codes: fetch_stock_basic_for_codes(codes, bs_client=baostock_client)
        )
    if stock_industry_fetch is None:
        stock_industry_fetch = (
            primary_fetchers.fetch_stock_industry
            if primary_fetchers is not None
            else lambda dates: fetch_stock_industry_for_dates(dates, bs_client=baostock_client)
        )
    if index_membership_fetch is None:
        index_membership_fetch = (
            primary_fetchers.fetch_index_membership
            if primary_fetchers is not None
            else lambda dates: fetch_index_membership_for_dates(dates, bs_client=baostock_client)
        )
    if financial_fetch is None:
        financial_fetch = (
            primary_fetchers.fetch_financial_tables
            if primary_fetchers is not None
            else lambda codes, year_quarters: fetch_quarterly_financial_tables(
                codes,
                year_quarters,
                bs_client=baostock_client,
            )
        )

    prefetched_index_membership_df = pd.DataFrame()
    needs_baostock_market_data = (
        daily_requires_baostock
        or benchmark_requires_baostock
        or (stock_universe_name == "hs300_zz500" and index_membership_requires_baostock)
    )
    try:
        with _baostock_session(enabled=needs_baostock_market_data):
            if stock_universe_name == "hs300_zz500":
                prefetched_index_membership_df = _load_cached_or_fetch_dated_frame(
                    paths["index_membership_path"],
                    lambda: index_membership_fetch(target_month_end_dates),
                    required_dates=target_month_end_dates,
                )
                if prediction_mode:
                    prefetched_index_membership_df = clip_dated_frame_as_of(
                        prefetched_index_membership_df,
                        as_of_date,
                    )
                stock_codes = _stock_codes_from_index_members(prefetched_index_membership_df)
                if not stock_codes:
                    raise ValueError("No HS300/ZZ500 index members found for target months")
                incidence_df = _filter_incidence_to_stock_universe(incidence_df, stock_codes)
            daily_df = _load_cached_or_fetch_market_frame(
                paths["daily_k_path"],
                lambda: daily_fetch(stock_codes, start_date, end_date),
                required_start_date=start_date,
                required_end_date=end_date,
            )
            daily_df = _filter_daily_to_stock_universe(daily_df, stock_codes)
            hs300_daily_df = _load_cached_or_fetch_market_frame(
                paths["hs300_daily_k_path"],
                lambda: _fetch_index_benchmark(
                    benchmark_fetch,
                    index_code=INDEX_CODE_HS300,
                    start_date=start_date,
                    end_date=end_date,
                    cache_path=paths["hs300_daily_k_path"],
                    use_cache=False,
                ),
                required_start_date=start_date,
                required_end_date=end_date,
            )
            zz500_daily_df = _load_cached_or_fetch_market_frame(
                paths["zz500_daily_k_path"],
                lambda: _fetch_index_benchmark(
                    benchmark_fetch,
                    index_code=INDEX_CODE_ZZ500,
                    start_date=start_date,
                    end_date=end_date,
                    cache_path=paths["zz500_daily_k_path"],
                    use_cache=False,
                ),
                required_start_date=start_date,
                required_end_date=end_date,
            )
    except Exception as exc:
        if not _should_use_tushare_fallback(exc):
            raise

        tushare_fetchers = _build_tushare_fetchers(tushare_token_url)
        daily_fetch = tushare_fetchers.fetch_daily_k
        benchmark_fetch = tushare_fetchers.fetch_benchmark
        stock_basic_fetch = tushare_fetchers.fetch_stock_basic
        stock_industry_fetch = tushare_fetchers.fetch_stock_industry
        index_membership_fetch = tushare_fetchers.fetch_index_membership
        financial_fetch = tushare_fetchers.fetch_financial_tables

        daily_requires_baostock = False
        benchmark_requires_baostock = False
        stock_basic_requires_baostock = False
        stock_industry_requires_baostock = False
        index_membership_requires_baostock = False
        financial_requires_baostock = False

        if stock_universe_name == "hs300_zz500":
            prefetched_index_membership_df = _load_cached_or_fetch_dated_frame(
                paths["index_membership_path"],
                lambda: index_membership_fetch(target_month_end_dates),
                required_dates=target_month_end_dates,
            )
            if prediction_mode:
                prefetched_index_membership_df = clip_dated_frame_as_of(
                    prefetched_index_membership_df,
                    as_of_date,
                )
            stock_codes = _stock_codes_from_index_members(prefetched_index_membership_df)
            if not stock_codes:
                raise ValueError("No HS300/ZZ500 index members found for target months")
            incidence_df = _filter_incidence_to_stock_universe(incidence_df, stock_codes)
        daily_df = _load_cached_or_fetch_market_frame(
            paths["daily_k_path"],
            lambda: daily_fetch(stock_codes, start_date, end_date),
            required_start_date=start_date,
            required_end_date=end_date,
        )
        daily_df = _filter_daily_to_stock_universe(daily_df, stock_codes)
        hs300_daily_df = _load_cached_or_fetch_market_frame(
            paths["hs300_daily_k_path"],
            lambda: _fetch_index_benchmark(
                benchmark_fetch,
                index_code=INDEX_CODE_HS300,
                start_date=start_date,
                end_date=end_date,
                cache_path=paths["hs300_daily_k_path"],
                use_cache=False,
            ),
            required_start_date=start_date,
            required_end_date=end_date,
        )
        zz500_daily_df = _load_cached_or_fetch_market_frame(
            paths["zz500_daily_k_path"],
            lambda: _fetch_index_benchmark(
                benchmark_fetch,
                index_code=INDEX_CODE_ZZ500,
                start_date=start_date,
                end_date=end_date,
                cache_path=paths["zz500_daily_k_path"],
                use_cache=False,
            ),
            required_start_date=start_date,
            required_end_date=end_date,
        )

    if prediction_mode:
        daily_df = clip_market_frame_as_of(daily_df, as_of_date)
        hs300_daily_df = clip_market_frame_as_of(hs300_daily_df, as_of_date)
        zz500_daily_df = clip_market_frame_as_of(zz500_daily_df, as_of_date)

    monthly_features_df = build_monthly_stock_features(daily_df, hs300_daily_df)
    labels_df = build_future_labels(
        daily_df,
        hs300_daily_df,
        required_horizons=(1,) if prediction_mode else FORWARD_RETURN_HORIZONS,
    )
    target_month_list = sorted(target_months)
    filtered_features_df = (
        monthly_features_df.loc[monthly_features_df["month"].isin(target_month_list)]
        .sort_values(["month", "code"], ignore_index=True)
    )
    filtered_labels_df = (
        labels_df.loc[labels_df["month"].isin(target_month_list)]
        .sort_values(["month", "code"], ignore_index=True)
    )
    benchmark_returns_df = build_monthly_dual_benchmark_returns(
        hs300_daily_df=hs300_daily_df,
        zz500_daily_df=zz500_daily_df,
        target_months=set(filtered_labels_df["month"].dropna().tolist()),
        required_horizons=(1,) if prediction_mode else FORWARD_RETURN_HORIZONS,
    )

    month_end_dates = _derive_month_end_dates(filtered_features_df)

    assert stock_basic_fetch is not None
    assert stock_industry_fetch is not None
    assert index_membership_fetch is not None

    try:
        with _baostock_session(
            enabled=stock_basic_requires_baostock
            or stock_industry_requires_baostock
            or index_membership_requires_baostock
        ):
            stock_basic_df = _load_cached_or_fetch_frame(
                paths["stock_basic_path"],
                lambda: stock_basic_fetch(stock_codes),
            )
            stock_industry_df = _load_cached_or_fetch_dated_frame(
                paths["stock_industry_path"],
                lambda: stock_industry_fetch(month_end_dates),
                required_dates=month_end_dates,
            )
            index_membership_df = (
                prefetched_index_membership_df
                if not prefetched_index_membership_df.empty
                else _load_cached_or_fetch_dated_frame(
                    paths["index_membership_path"],
                    lambda: index_membership_fetch(month_end_dates),
                    required_dates=month_end_dates,
                )
            )
    except Exception as exc:
        if not _should_use_tushare_fallback(exc):
            raise

        tushare_fetchers = _build_tushare_fetchers(tushare_token_url)
        stock_basic_fetch = tushare_fetchers.fetch_stock_basic
        stock_industry_fetch = tushare_fetchers.fetch_stock_industry
        index_membership_fetch = tushare_fetchers.fetch_index_membership
        financial_fetch = tushare_fetchers.fetch_financial_tables
        financial_requires_baostock = False

        stock_basic_df = _load_cached_or_fetch_frame(
            paths["stock_basic_path"],
            lambda: stock_basic_fetch(stock_codes),
        )
        stock_industry_df = _load_cached_or_fetch_dated_frame(
            paths["stock_industry_path"],
            lambda: stock_industry_fetch(month_end_dates),
            required_dates=month_end_dates,
        )
        index_membership_df = (
            prefetched_index_membership_df
            if not prefetched_index_membership_df.empty
            else _load_cached_or_fetch_dated_frame(
                paths["index_membership_path"],
                lambda: index_membership_fetch(month_end_dates),
                required_dates=month_end_dates,
                )
            )

    if prediction_mode:
        stock_industry_df = clip_dated_frame_as_of(stock_industry_df, as_of_date)
        index_membership_df = clip_dated_frame_as_of(index_membership_df, as_of_date)

    filtered_features_df = build_monthly_stock_metadata_features(
        filtered_features_df,
        stock_basic_df,
        stock_industry_df,
        index_membership_df,
    )
    assert financial_fetch is not None
    year_quarters = _derive_financial_year_quarters(
        target_month_list,
        as_of_date=as_of_date if prediction_mode else None,
    )
    try:
        with _baostock_session(enabled=financial_requires_baostock):
            financial_tables = _load_cached_or_fetch_financial_tables(
                paths,
                lambda: financial_fetch(stock_codes, year_quarters),
            )
    except Exception as exc:
        if not _should_use_tushare_fallback(exc):
            raise

        tushare_fetchers = _build_tushare_fetchers(tushare_token_url)
        financial_fetch = tushare_fetchers.fetch_financial_tables
        financial_tables = _load_cached_or_fetch_financial_tables(
            paths,
            lambda: financial_fetch(stock_codes, year_quarters),
        )

    if prediction_mode:
        financial_tables = clip_financial_tables_as_of(financial_tables, as_of_date)

    filtered_features_df = build_monthly_financial_features(
        filtered_features_df,
        financial_tables,
    )

    alpha_results_dir = Path(
        alpha_factor_results_dir or (root_path / "alpha" / "results" / "all_alpha_daily")
    )
    if not alpha_results_dir.is_absolute():
        alpha_results_dir = root_path / alpha_results_dir
    filtered_features_df, alpha_audit_df, alpha_manifest = build_monthly_alpha_features(
        daily_df=daily_df,
        monthly_features_df=filtered_features_df,
        results_dir=alpha_results_dir,
        top_n=alpha_factor_top_n,
    )

    if filtered_features_df.empty:
        raise ValueError(f"No monthly stock features available for months: {target_month_list}")
    if filtered_labels_df.empty and not prediction_mode:
        raise ValueError(f"No monthly labels available for months: {target_month_list}")
    if benchmark_returns_df.empty and not prediction_mode:
        raise ValueError(f"No benchmark returns available for months: {target_month_list}")

    _write_parquet(reports_df, paths["reports_path"])
    _write_parquet(incidence_df, paths["incidence_path"])
    _write_parquet(daily_df, paths["daily_k_path"])
    _write_parquet(hs300_daily_df, paths["hs300_daily_k_path"])
    _write_parquet(zz500_daily_df, paths["zz500_daily_k_path"])
    _write_parquet(stock_basic_df, paths["stock_basic_path"])
    _write_parquet(stock_industry_df, paths["stock_industry_path"])
    _write_parquet(index_membership_df, paths["index_membership_path"])
    for table_name in FINANCIAL_QUERY_METHODS:
        _write_parquet(
            financial_tables.get(table_name, pd.DataFrame()),
            paths[f"financial_{table_name}_path"],
        )
    _write_parquet(filtered_features_df, paths["monthly_features_path"])
    _write_parquet(alpha_audit_df, paths["alpha_audit_path"])
    write_alpha_manifest(alpha_manifest, paths["alpha_manifest_path"])
    _write_parquet(filtered_labels_df, paths["labels_path"])
    _write_parquet(benchmark_returns_df, paths["benchmark_path"])
    return paths


def _derive_month_end_dates(monthly_features_df: pd.DataFrame) -> list[str]:
    if monthly_features_df.empty:
        return []
    month_dates = (
        monthly_features_df[["month", "date"]]
        .drop_duplicates(subset=["month"])
        .sort_values("month")
    )
    return pd.to_datetime(month_dates["date"]).dt.strftime("%Y-%m-%d").tolist()


def clip_market_frame_as_of(
    frame: pd.DataFrame,
    as_of_date: str | None,
) -> pd.DataFrame:
    if frame.empty or as_of_date is None:
        return frame.copy()
    resolved_as_of_date = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(resolved_as_of_date):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    if "date" not in frame.columns:
        raise ValueError("Market data must contain a date column")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame.loc[dates <= resolved_as_of_date].reset_index(drop=True)


def clip_dated_frame_as_of(
    frame: pd.DataFrame,
    as_of_date: str | None,
) -> pd.DataFrame:
    if frame.empty or as_of_date is None:
        return frame.copy()
    resolved_as_of_date = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(resolved_as_of_date):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    if "date" not in frame.columns:
        raise ValueError("Dated metadata must contain a date column")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    return frame.loc[dates <= resolved_as_of_date].reset_index(drop=True)


def clip_financial_tables_as_of(
    financial_tables: dict[str, pd.DataFrame],
    as_of_date: str | None,
) -> dict[str, pd.DataFrame]:
    if as_of_date is None:
        return {name: frame.copy() for name, frame in financial_tables.items()}
    resolved_as_of_date = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(resolved_as_of_date):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    clipped: dict[str, pd.DataFrame] = {}
    for table_name, frame in financial_tables.items():
        if frame.empty:
            clipped[table_name] = frame.copy()
            continue
        if "pubDate" not in frame.columns:
            raise ValueError(f"Financial table {table_name} must contain a pubDate column")
        publish_dates = pd.to_datetime(frame["pubDate"], errors="coerce")
        clipped[table_name] = frame.loc[
            publish_dates.notna() & (publish_dates <= resolved_as_of_date)
        ].reset_index(drop=True)
    return clipped


def clip_reports_for_prediction(
    reports_df: pd.DataFrame,
    data_start_month: str,
    signal_month: str,
    as_of_date: str,
) -> pd.DataFrame:
    if reports_df.empty:
        return reports_df.copy()
    required_columns = {"report_month", "publish_date"}
    missing_columns = required_columns - set(reports_df.columns)
    if missing_columns:
        raise ValueError(
            f"Prediction reports missing required columns: {sorted(missing_columns)}"
        )
    target_months = derive_target_months(data_start_month, signal_month)
    resolved_as_of_date = pd.to_datetime(as_of_date, errors="coerce")
    if pd.isna(resolved_as_of_date):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    report_months = reports_df["report_month"].map(_normalize_report_month)
    publish_dates = pd.to_datetime(reports_df["publish_date"], errors="coerce")
    return reports_df.loc[
        report_months.isin(target_months) & (publish_dates <= resolved_as_of_date)
    ].reset_index(drop=True)


def _derive_financial_year_quarters(
    target_months: list[str],
    as_of_date: str | None = None,
) -> list[tuple[int, int]]:
    years = [int(str(month).split("-")[0]) for month in target_months if str(month)]
    if not years:
        return []
    start_year = min(years) - 2
    end_year = max(years)
    resolved_as_of_date = (
        pd.to_datetime(as_of_date, errors="coerce") if as_of_date is not None else None
    )
    if resolved_as_of_date is not None and pd.isna(resolved_as_of_date):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    completed_quarter = None
    if resolved_as_of_date is not None:
        current_quarter = resolved_as_of_date.to_period("Q")
        completed_quarter = (
            current_quarter
            if resolved_as_of_date.normalize() >= current_quarter.end_time.normalize()
            else current_quarter - 1
        )
    return [
        (year, quarter)
        for year in range(start_year, end_year + 1)
        for quarter in (1, 2, 3, 4)
        if resolved_as_of_date is None
        or pd.Period(f"{year}Q{quarter}", freq="Q") <= completed_quarter
    ]


def _fetch_index_benchmark(
    fetch_fn: Callable[..., pd.DataFrame],
    index_code: str,
    start_date: str,
    end_date: str,
    cache_path: Path,
    use_cache: bool = True,
) -> pd.DataFrame:
    if not use_cache:
        cache_path = cache_path.with_name(f".{cache_path.name}.refresh")
        try:
            return fetch_fn(
                index_code=index_code,
                start_date=start_date,
                end_date=end_date,
                cache_path=cache_path,
            )
        finally:
            if cache_path.exists():
                cache_path.unlink()
    return fetch_fn(
        index_code=index_code,
        start_date=start_date,
        end_date=end_date,
        cache_path=cache_path,
    )


def _build_artifact_paths(target_artifact_dir: Path) -> dict[str, Path]:
    paths = {
        "artifact_dir": target_artifact_dir,
        "reports_path": target_artifact_dir / "reports.parquet",
        "incidence_path": target_artifact_dir / "incidence.parquet",
        "daily_k_path": target_artifact_dir / "daily_k.parquet",
        "hs300_daily_k_path": target_artifact_dir / "hs300_daily_k.parquet",
        "zz500_daily_k_path": target_artifact_dir / "zz500_daily_k.parquet",
        "benchmark_daily_k_path": target_artifact_dir / "hs300_daily_k.parquet",
        "monthly_features_path": target_artifact_dir / "monthly_stock_features.parquet",
        "labels_path": target_artifact_dir / "monthly_labels.parquet",
        "benchmark_path": target_artifact_dir / "monthly_benchmark_returns.parquet",
        "stock_basic_path": target_artifact_dir / "stock_basic.parquet",
        "stock_industry_path": target_artifact_dir / "stock_industry.parquet",
        "index_membership_path": target_artifact_dir / "index_membership.parquet",
        "alpha_audit_path": target_artifact_dir / "alpha_factor_audit.parquet",
        "alpha_manifest_path": target_artifact_dir / "alpha_factor_manifest.json",
    }
    for table_name in FINANCIAL_QUERY_METHODS:
        paths[f"financial_{table_name}_path"] = (
            target_artifact_dir / f"financial_{table_name}.parquet"
        )
    return paths


def _call_parse_reports(
    parse_reports_fn: Callable[..., pd.DataFrame],
    workspace_root: Path,
    markdown_path: Path,
    start_month: str | None = None,
    end_month: str | None = None,
) -> pd.DataFrame:
    try:
        return parse_reports_fn(
            workspace_root,
            markdown_path,
            start_month=start_month,
            end_month=end_month,
        )
    except TypeError:
        return parse_reports_fn(workspace_root, markdown_path)


def _load_cached_or_fetch_frame(
    cache_path: Path,
    fetch_fn: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return fetch_fn()


def _load_cached_or_fetch_dated_frame(
    cache_path: Path,
    fetch_fn: Callable[[], pd.DataFrame],
    required_dates: list[str],
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if _dated_frame_covers_months(cached, required_dates):
            return cached
    return fetch_fn()


def _dated_cache_path_covers_months(cache_path: Path, required_dates: list[str]) -> bool:
    if not cache_path.exists():
        return False
    return _dated_frame_covers_months(pd.read_parquet(cache_path), required_dates)


def _dated_frame_covers_months(frame: pd.DataFrame, required_dates: list[str]) -> bool:
    if frame.empty or "date" not in frame.columns:
        return False
    cached_dates = set(
        pd.to_datetime(frame["date"], errors="coerce").dropna().dt.normalize()
    )
    required_timestamps = set(
        pd.to_datetime(pd.Series(required_dates), errors="coerce").dropna().dt.normalize()
    )
    return bool(required_timestamps) and required_timestamps.issubset(cached_dates)


def _load_cached_or_fetch_market_frame(
    cache_path: Path,
    fetch_fn: Callable[[], pd.DataFrame],
    required_end_date: str,
    required_start_date: str | None = None,
) -> pd.DataFrame:
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if _market_frame_covers_date_range(
            cached,
            required_start_date=required_start_date,
            required_end_date=required_end_date,
        ):
            return cached
    return fetch_fn()


def _market_frame_covers_date_range(
    frame: pd.DataFrame,
    required_start_date: str | None,
    required_end_date: str,
) -> bool:
    if frame.empty or "date" not in frame.columns:
        return False
    dates = pd.to_datetime(frame["date"], errors="coerce")
    min_date = dates.min()
    max_date = dates.max()
    required_start = pd.to_datetime(required_start_date, errors="coerce")
    required = pd.to_datetime(required_end_date, errors="coerce")
    if pd.isna(max_date) or pd.isna(required):
        return False
    if required_start_date is not None and (
        pd.isna(min_date)
        or pd.isna(required_start)
        or min_date.normalize() > required_start.normalize()
    ):
        return False
    return max_date.normalize() >= required.normalize()


def _all_financial_artifacts_exist(paths: dict[str, Path]) -> bool:
    return all(
        paths[f"financial_{table_name}_path"].exists()
        for table_name in FINANCIAL_QUERY_METHODS
    )


def _load_cached_or_fetch_financial_tables(
    paths: dict[str, Path],
    fetch_fn: Callable[[], dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    if _all_financial_artifacts_exist(paths):
        return {
            table_name: pd.read_parquet(paths[f"financial_{table_name}_path"])
            for table_name in FINANCIAL_QUERY_METHODS
        }
    return fetch_fn()


def _should_use_tushare_fallback(exc: Exception) -> bool:
    # Tushare 可能需要付费权限，仅在用户显式开启时作为回退源。
    if not _is_truthy(os.getenv("HYPERGRAPH_ENABLE_TUSHARE_FALLBACK", "0")):
        return False
    message = str(exc).lower()
    # Baostock 偶尔会在 login 返回成功后，让首个查询仍报会话未登录。
    baostock_session_failed = (
        "baostock login failed" in message
        or "10001001" in message
        or "you don't login" in message
    )
    return baostock_session_failed or _is_timeout_error(exc)


def _build_tushare_fetchers(tushare_token_url: str | None):
    try:
        return build_tushare_fetchers(token_url=tushare_token_url)
    except TypeError as exc:
        if "token_url" not in str(exc):
            raise
        return build_tushare_fetchers()


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    message = str(exc).lower()
    return (
        "timed out" in message
        or "timeout" in message
        or "超时" in message
    )


@contextmanager
def _baostock_session(
    enabled: bool,
    max_retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> Iterator[None]:
    if not enabled or baostock_client is None:
        yield
        return

    previous_socket_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(BAOSTOCK_REQUEST_TIMEOUT_SECONDS)
    try:
        attempts = max(1, int(max_retries))
        last_error_msg = ""
        for attempt in range(1, attempts + 1):
            try:
                login_result = baostock_client.login()
            except Exception as exc:
                if _is_timeout_error(exc):
                    raise RuntimeError(
                        f"baostock login timed out after {BAOSTOCK_REQUEST_TIMEOUT_SECONDS:.1f}s: {exc}"
                    ) from exc
                last_error_msg = str(exc)
            else:
                if login_result.error_code == "0":
                    break
                last_error_msg = str(login_result.error_msg)

            try:
                baostock_client.logout()
            except Exception:
                pass
            if attempt < attempts and retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
        else:
            raise RuntimeError(
                f"baostock login failed after {attempts} attempts: {last_error_msg}"
            )

        try:
            yield
        finally:
            baostock_client.logout()
    finally:
        socket.setdefaulttimeout(previous_socket_timeout)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_stock_universe(value: str) -> str:
    normalized = str(value or "hs300_zz500").strip().lower()
    if normalized not in {"report_only", "hs300_zz500"}:
        raise ValueError("stock_universe must be one of: report_only, hs300_zz500")
    return normalized


def _derive_target_month_end_dates(
    target_months: list[str],
    as_of_date: str | None = None,
) -> list[str]:
    resolved_as_of_date = (
        pd.to_datetime(as_of_date, errors="coerce") if as_of_date is not None else None
    )
    if resolved_as_of_date is not None and pd.isna(resolved_as_of_date):
        raise ValueError(f"Invalid as_of_date: {as_of_date}")
    dates: list[str] = []
    for month in target_months:
        if not _normalize_report_month(month):
            continue
        month_end = pd.Period(month, freq="M").end_time.normalize()
        if resolved_as_of_date is not None and month_end > resolved_as_of_date:
            month_end = resolved_as_of_date.normalize()
        dates.append(month_end.date().isoformat())
    return dates


def _stock_codes_from_index_members(index_membership_df: pd.DataFrame) -> list[str]:
    codes: set[str] = set()
    if not index_membership_df.empty and {"code", "index_name"}.issubset(index_membership_df.columns):
        members = index_membership_df.loc[
            index_membership_df["index_name"].astype(str).str.lower().isin({"hs300", "zz500"}),
            "code",
        ]
        codes.update(normalize_baostock_code(code) for code in members.dropna().astype(str))
    return sorted(codes)


def _filter_incidence_to_stock_universe(
    incidence_df: pd.DataFrame,
    stock_codes: list[str],
) -> pd.DataFrame:
    if incidence_df.empty or "stock_code" not in incidence_df.columns:
        return incidence_df.copy()
    universe_codes = {normalize_baostock_code(code) for code in stock_codes}
    filtered = incidence_df.copy()
    normalized_stock_codes = filtered["stock_code"].map(normalize_baostock_code)
    return filtered.loc[normalized_stock_codes.isin(universe_codes)].reset_index(drop=True)


def _filter_daily_to_stock_universe(
    daily_df: pd.DataFrame,
    stock_codes: list[str],
) -> pd.DataFrame:
    if daily_df.empty or "code" not in daily_df.columns:
        return daily_df.copy()
    universe_codes = {normalize_baostock_code(code) for code in stock_codes}
    filtered = daily_df.copy()
    normalized_codes = filtered["code"].map(normalize_baostock_code)
    return filtered.loc[normalized_codes.isin(universe_codes)].reset_index(drop=True)


def _normalize_report_month(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    try:
        return str(pd.Period(text, freq="M"))
    except Exception:
        match = REPORT_MONTH_RE.search(text)
        if not match:
            return ""
        year = int(match.group("year"))
        month = int(match.group("month"))
        if month < 1 or month > 12:
            return ""
        return f"{year:04d}-{month:02d}"


def _month_from_dataset_name(name: str) -> str:
    text = str(name).strip()
    match = re.match(r"^(?P<year>\d{4})(?P<month>\d{2})(?:\D|$)", text)
    if not match:
        return ""
    month = int(match.group("month"))
    if month < 1 or month > 12:
        return ""
    return f"{int(match.group('year')):04d}-{month:02d}"


def _dataset_name_from_markdown_dir(
    markdown_dir: str | Path,
    start_month: str | None = None,
    end_month: str | None = None,
) -> str:
    text = Path(markdown_dir).name
    month = _month_from_dataset_name(text)
    if month:
        return text
    resolved_start = _normalize_report_month(start_month)
    resolved_end = _normalize_report_month(end_month)
    if resolved_start and resolved_end:
        return f"{resolved_start.replace('-', '')}_{resolved_end.replace('-', '')}"
    return text


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare month-scoped alphagraph training artifacts",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--markdown-dir", type=str, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--start-month", type=str, default=None)
    parser.add_argument("--end-month", type=str, default=None)
    parser.add_argument("--data-start-month", type=str, default=None)
    parser.add_argument("--signal-month", type=str, default=None)
    parser.add_argument("--as-of-date", type=str, default=None)
    parser.add_argument("--prediction", action="store_true")
    parser.add_argument("--stock-universe", type=str, default=os.getenv("STOCK_UNIVERSE", "hs300_zz500"))
    parser.add_argument("--market-data-source", type=str, default=os.getenv("MARKET_DATA_SOURCE", "akshare"))
    parser.add_argument("--tushare-token-url", type=str, default=os.getenv("TUSHARE_TOKEN_URL"))
    parser.add_argument(
        "--alpha-factor-results-dir",
        type=Path,
        default=os.getenv("ALPHA_FACTOR_RESULTS_DIR"),
    )
    parser.add_argument(
        "--alpha-factor-top-n",
        type=int,
        default=int(os.getenv("ALPHA_FACTOR_TOP_N", "10")),
    )
    args = parser.parse_args(argv)

    paths = prepare_monthly_dataset(
        workspace_root=args.workspace_root,
        markdown_dir=args.markdown_dir,
        artifact_dir=args.artifact_dir,
        start_month=args.start_month,
        end_month=args.end_month,
        stock_universe=args.stock_universe,
        market_data_source=args.market_data_source,
        tushare_token_url=args.tushare_token_url,
        data_start_month=args.data_start_month,
        signal_month=args.signal_month,
        as_of_date=args.as_of_date,
        prediction_mode=args.prediction,
        alpha_factor_results_dir=args.alpha_factor_results_dir,
        alpha_factor_top_n=args.alpha_factor_top_n,
    )
    payload: dict[str, Any] = {key: str(value) for key, value in paths.items()}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
