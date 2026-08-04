from __future__ import annotations

import importlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests

from alphagraph.data.financial_features import FINANCIAL_QUERY_METHODS
from alphagraph.data.market_data import (
    INDEX_CODE_HS300,
    INDEX_CODE_ZZ500,
    normalize_baostock_code,
)

# Credentials must be supplied through the environment or function arguments.
DEFAULT_TUSHARE_TOKEN = ""
TOKEN_URL = ""
STOCK_BASIC_FIELDS = (
    "ts_code,symbol,name,area,industry,fullname,enname,cnspell,market,exchange,"
    "curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type"
)

FINA_INDICATOR_FIELDS = (
    "ts_code,ann_date,end_date,roe_avg,roe,netprofit_margin,grossprofit_margin,"
    "netprofit_yoy,assets_yoy,ar_turn,inv_turn,current_ratio,quick_ratio,ocf_to_or"
)

INDEX_DAILY_CODE_CANDIDATES = {
    INDEX_CODE_HS300: ("000300.SH", "399300.SZ"),
    INDEX_CODE_ZZ500: ("000905.SH", "399905.SZ"),
    "sh.000016": ("000016.SH", "399016.SZ"),
}

INDEX_WEIGHT_CODE_CANDIDATES = {
    "hs300": ("000300.SH", "399300.SZ"),
    "zz500": ("000905.SH", "399905.SZ"),
    "sz50": ("000016.SH", "399016.SZ"),
}


@dataclass
class TushareFetchers:
    fetch_daily_k: Callable[[list[str], str, str], pd.DataFrame]
    fetch_benchmark: Callable[..., pd.DataFrame]
    fetch_stock_basic: Callable[[list[str]], pd.DataFrame]
    fetch_stock_industry: Callable[[list[str]], pd.DataFrame]
    fetch_index_membership: Callable[[list[str]], pd.DataFrame]
    fetch_financial_tables: Callable[[list[str], list[tuple[int, int]]], dict[str, pd.DataFrame]]


def build_tushare_fetchers(
    token: str | None = None,
    token_url: str | None = None,
) -> TushareFetchers:
    pro = _create_tushare_pro(token=token, token_url=token_url)
    stock_basic_cache: pd.DataFrame | None = None
    index_weight_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def fetch_daily_k(codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        ts_codes = sorted({baostock_to_tushare_code(code) for code in codes if str(code).strip()})
        if not ts_codes:
            return _empty_daily_k_frame()

        start_key = _to_tushare_date(start_date)
        end_key = _to_tushare_date(end_date)

        frames: list[pd.DataFrame] = []
        for ts_code in ts_codes:
            frame = _call_tushare(
                pro.daily,
                strict=False,
                ts_code=ts_code,
                start_date=start_key,
                end_date=end_key,
            )
            if frame.empty:
                continue
            frames.append(frame)
            _maybe_sleep()

        if not frames:
            raise RuntimeError(
                f"Tushare daily returned empty data for {len(ts_codes)} symbols in [{start_key}, {end_key}]"
            )

        daily = pd.concat(frames, ignore_index=True)
        return _normalize_tushare_daily(daily)

    def fetch_benchmark(
        index_code: str,
        start_date: str,
        end_date: str,
        cache_path: Path,
    ) -> pd.DataFrame:
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        start_key = _to_tushare_date(start_date)
        end_key = _to_tushare_date(end_date)
        candidates = _index_code_candidates(index_code)
        for ts_index_code in candidates:
            frame = _call_tushare(
                pro.index_daily,
                strict=False,
                ts_code=ts_index_code,
                start_date=start_key,
                end_date=end_key,
            )
            if frame.empty:
                continue
            normalized = _normalize_tushare_index_daily(frame, index_code=index_code)
            normalized.to_parquet(cache_path, index=False)
            return normalized

        raise RuntimeError(
            f"Tushare index_daily returned empty data for {index_code} in [{start_key}, {end_key}]"
        )

    def fetch_stock_basic(codes: list[str]) -> pd.DataFrame:
        nonlocal stock_basic_cache
        if stock_basic_cache is None:
            stock_basic_cache = _fetch_stock_basic_all(pro)

        normalized_codes = {normalize_baostock_code(code) for code in codes if str(code).strip()}
        basic = stock_basic_cache
        if normalized_codes:
            basic = basic.loc[basic["code"].isin(sorted(normalized_codes))].copy()
        return basic.reset_index(drop=True)

    def fetch_stock_industry(dates: list[str]) -> pd.DataFrame:
        nonlocal stock_basic_cache
        if stock_basic_cache is None:
            stock_basic_cache = _fetch_stock_basic_all(pro)
        if stock_basic_cache.empty:
            return pd.DataFrame(columns=["date", "code", "industry", "industryClassification"])

        unique_dates = sorted({_to_iso_date(value) for value in dates if str(value).strip()})
        if not unique_dates:
            return pd.DataFrame(columns=["date", "code", "industry", "industryClassification"])

        industry_base = stock_basic_cache[["code", "industry", "industryClassification"]].copy()
        industry_base = industry_base.drop_duplicates(subset=["code"], keep="last")
        date_frame = pd.DataFrame({"date": unique_dates})
        date_frame["_key"] = 1
        industry_base["_key"] = 1
        merged = date_frame.merge(industry_base, on="_key", how="inner").drop(columns=["_key"])
        return merged[["date", "code", "industry", "industryClassification"]]

    def fetch_index_membership(dates: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        unique_dates = sorted({_to_iso_date(value) for value in dates if str(value).strip()})
        for date_text in unique_dates:
            target_date = pd.Timestamp(date_text)
            month_key = target_date.strftime("%Y%m")
            start_key = target_date.replace(day=1).strftime("%Y%m%d")
            end_key = (target_date.replace(day=1) + pd.offsets.MonthEnd(1)).strftime("%Y%m%d")

            for index_name, candidates in INDEX_WEIGHT_CODE_CANDIDATES.items():
                index_month_df = pd.DataFrame()
                for candidate in candidates:
                    cache_key = (candidate, month_key)
                    if cache_key not in index_weight_cache:
                        frame = _call_tushare(
                            pro.index_weight,
                            strict=False,
                            index_code=candidate,
                            start_date=start_key,
                            end_date=end_key,
                        )
                        index_weight_cache[cache_key] = frame
                        _maybe_sleep()
                    frame = index_weight_cache[cache_key]
                    if frame.empty:
                        continue
                    index_month_df = frame.copy()
                    break

                if index_month_df.empty or "trade_date" not in index_month_df.columns:
                    continue

                index_month_df["trade_date"] = pd.to_datetime(
                    index_month_df["trade_date"], format="%Y%m%d", errors="coerce"
                )
                index_month_df = index_month_df.dropna(subset=["trade_date"])
                if index_month_df.empty:
                    continue

                visible = index_month_df.loc[index_month_df["trade_date"] <= target_date]
                if visible.empty:
                    visible = index_month_df
                latest_date = visible["trade_date"].max()
                latest_members = visible.loc[visible["trade_date"] == latest_date]

                for con_code in latest_members.get("con_code", pd.Series(dtype=str)).dropna().astype(str):
                    rows.append(
                        {
                            "date": date_text,
                            "code": tushare_to_baostock_code(con_code),
                            "index_name": index_name,
                        }
                    )

        if not rows:
            return pd.DataFrame(columns=["date", "code", "index_name"])
        return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)

    def fetch_financial_tables(
        codes: list[str],
        year_quarters: list[tuple[int, int]],
    ) -> dict[str, pd.DataFrame]:
        if not codes or not year_quarters:
            return _empty_financial_tables()

        start_key, end_key = _financial_date_range(year_quarters)
        ts_codes = sorted({baostock_to_tushare_code(code) for code in codes if str(code).strip()})

        frames: list[pd.DataFrame] = []
        for ts_code in ts_codes:
            frame = _call_tushare(
                pro.fina_indicator,
                strict=False,
                ts_code=ts_code,
                start_date=start_key,
                end_date=end_key,
                fields=FINA_INDICATOR_FIELDS,
            )
            if frame.empty:
                frame = _call_tushare(
                    pro.fina_indicator,
                    strict=False,
                    ts_code=ts_code,
                    start_date=start_key,
                    end_date=end_key,
                )
            if frame.empty:
                continue
            frames.append(frame)
            _maybe_sleep()

        if not frames:
            return _empty_financial_tables()

        indicators = pd.concat(frames, ignore_index=True)
        financial = _map_fina_indicator_to_financial_tables(indicators)
        if not financial:
            return _empty_financial_tables()
        return financial

    return TushareFetchers(
        fetch_daily_k=fetch_daily_k,
        fetch_benchmark=fetch_benchmark,
        fetch_stock_basic=fetch_stock_basic,
        fetch_stock_industry=fetch_stock_industry,
        fetch_index_membership=fetch_index_membership,
        fetch_financial_tables=fetch_financial_tables,
    )


def baostock_to_tushare_code(code: str) -> str:
    normalized = normalize_baostock_code(code)
    if "." not in normalized:
        return normalized
    market, symbol = normalized.split(".", 1)
    return f"{symbol.upper()}.{market.upper()}"


def tushare_to_baostock_code(code: str) -> str:
    raw = str(code).strip().upper()
    if "." not in raw:
        return normalize_baostock_code(raw)
    symbol, market = raw.split(".", 1)
    if market in {"SH", "SSE"}:
        return f"sh.{symbol.lower()}"
    if market in {"SZ", "SZSE"}:
        return f"sz.{symbol.lower()}"
    return normalize_baostock_code(symbol)


def _create_tushare_pro(token: str | None, token_url: str | None = None) -> Any:
    try:
        tushare_module = importlib.import_module("tushare")
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("tushare is required for fallback mode. Please install tushare.") from exc

    use_token = (
        token
        or os.getenv("TUSHARE_TOKEN")
        or _fetch_tushare_token_from_url(token_url or os.getenv("TUSHARE_TOKEN_URL") or TOKEN_URL)
        or DEFAULT_TUSHARE_TOKEN
    ).strip()
    if not use_token:
        raise RuntimeError("Missing TUSHARE token. Set TUSHARE_TOKEN or TUSHARE_TOKEN_URL.")

    pro = tushare_module.pro_api(token=use_token)
    return pro


def _fetch_tushare_token_from_url(token_url: str | None) -> str:
    if not token_url:
        return ""
    try:
        return requests.get(token_url, timeout=20).text.strip()
    except Exception:
        return ""


def _fetch_stock_basic_all(pro: Any) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for list_status in ("L", "D", "P"):
        frame = _call_tushare(
            pro.stock_basic,
            strict=False,
            exchange="",
            list_status=list_status,
            fields=STOCK_BASIC_FIELDS,
        )
        if frame.empty:
            continue
        frames.append(frame)
        _maybe_sleep()

    if not frames:
        return pd.DataFrame(
            columns=[
                "code",
                "ipoDate",
                "type",
                "status",
                "outDate",
                "industry",
                "industryClassification",
            ]
        )

    basic = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ts_code"], keep="last")
    basic["code"] = basic["ts_code"].map(tushare_to_baostock_code)
    basic["ipoDate"] = basic.get("list_date", pd.Series(dtype=str)).map(_to_iso_date)
    basic["outDate"] = basic.get("delist_date", pd.Series(dtype=str)).map(_to_iso_date)
    basic["industry"] = basic.get("industry", pd.Series(dtype=str)).fillna("").astype(str)
    basic["industryClassification"] = "tushare_stock_basic"
    basic["status"] = basic.get("list_status", pd.Series(dtype=str)).map(
        {"L": "1", "D": "0", "P": "0", "G": "0"}
    ).fillna("0")
    basic["type"] = "1"

    return basic[
        [
            "code",
            "ipoDate",
            "type",
            "status",
            "outDate",
            "industry",
            "industryClassification",
        ]
    ].reset_index(drop=True)


def _normalize_tushare_daily(frame: pd.DataFrame) -> pd.DataFrame:
    daily = frame.copy()
    if daily.empty:
        return _empty_daily_k_frame()

    daily["date"] = pd.to_datetime(daily.get("trade_date"), format="%Y%m%d", errors="coerce")
    daily["code"] = daily.get("ts_code", pd.Series(dtype=str)).map(tushare_to_baostock_code)

    daily["open"] = pd.to_numeric(daily.get("open"), errors="coerce")
    daily["high"] = pd.to_numeric(daily.get("high"), errors="coerce")
    daily["low"] = pd.to_numeric(daily.get("low"), errors="coerce")
    daily["close"] = pd.to_numeric(daily.get("close"), errors="coerce")
    daily["preclose"] = pd.to_numeric(daily.get("pre_close"), errors="coerce")
    daily["volume"] = pd.to_numeric(daily.get("vol"), errors="coerce")
    daily["amount"] = pd.to_numeric(daily.get("amount"), errors="coerce")
    daily["tradestatus"] = 1
    daily["turn"] = 0.0
    daily["pctChg"] = pd.to_numeric(daily.get("pct_chg"), errors="coerce")
    daily["peTTM"] = 0.0
    daily["pbMRQ"] = 0.0
    daily["psTTM"] = 0.0
    daily["pcfNcfTTM"] = 0.0
    daily["isST"] = 0

    normalized = daily[
        [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "tradestatus",
            "turn",
            "pctChg",
            "peTTM",
            "pbMRQ",
            "psTTM",
            "pcfNcfTTM",
            "isST",
        ]
    ].dropna(subset=["date", "code"])
    return normalized.sort_values(["code", "date"], ignore_index=True)


def _normalize_tushare_index_daily(frame: pd.DataFrame, index_code: str) -> pd.DataFrame:
    daily = frame.copy()
    if daily.empty:
        return _empty_daily_k_frame()

    daily["date"] = pd.to_datetime(daily.get("trade_date"), format="%Y%m%d", errors="coerce")
    daily["code"] = normalize_baostock_code(index_code)

    daily["open"] = pd.to_numeric(daily.get("open"), errors="coerce")
    daily["high"] = pd.to_numeric(daily.get("high"), errors="coerce")
    daily["low"] = pd.to_numeric(daily.get("low"), errors="coerce")
    daily["close"] = pd.to_numeric(daily.get("close"), errors="coerce")
    daily["preclose"] = pd.to_numeric(daily.get("pre_close"), errors="coerce")
    daily["volume"] = pd.to_numeric(daily.get("vol"), errors="coerce")
    daily["amount"] = pd.to_numeric(daily.get("amount"), errors="coerce")
    daily["tradestatus"] = 1
    daily["turn"] = 0.0
    daily["pctChg"] = pd.to_numeric(daily.get("pct_chg"), errors="coerce")
    daily["peTTM"] = 0.0
    daily["pbMRQ"] = 0.0
    daily["psTTM"] = 0.0
    daily["pcfNcfTTM"] = 0.0
    daily["isST"] = 0

    normalized = daily[
        [
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "tradestatus",
            "turn",
            "pctChg",
            "peTTM",
            "pbMRQ",
            "psTTM",
            "pcfNcfTTM",
            "isST",
        ]
    ].dropna(subset=["date"])
    return normalized.sort_values(["code", "date"], ignore_index=True)


def _map_fina_indicator_to_financial_tables(indicators: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if indicators.empty:
        return _empty_financial_tables()

    frame = indicators.copy()
    frame["code"] = frame.get("ts_code", pd.Series(dtype=str)).map(tushare_to_baostock_code)
    frame["pubDate"] = frame.get("ann_date", pd.Series(dtype=str)).map(_to_iso_date)
    frame["statDate"] = frame.get("end_date", pd.Series(dtype=str)).map(_to_iso_date)

    roe_avg = _to_numeric_series(frame, "roe_avg")
    roe = _to_numeric_series(frame, "roe")

    frame["profit_roeAvg"] = roe_avg.fillna(roe).fillna(0.0)
    frame["profit_npMargin"] = _to_numeric_series(frame, "netprofit_margin").fillna(0.0)
    frame["profit_gpMargin"] = _to_numeric_series(frame, "grossprofit_margin").fillna(0.0)
    frame["growth_YOYNI"] = _to_numeric_series(frame, "netprofit_yoy").fillna(0.0)
    frame["growth_YOYAsset"] = _to_numeric_series(frame, "assets_yoy").fillna(0.0)
    frame["operation_NRTurnRatio"] = _to_numeric_series(frame, "ar_turn").fillna(0.0)
    frame["operation_INVTurnRatio"] = _to_numeric_series(frame, "inv_turn").fillna(0.0)
    frame["balance_currentRatio"] = _to_numeric_series(frame, "current_ratio").fillna(0.0)
    frame["balance_quickRatio"] = _to_numeric_series(frame, "quick_ratio").fillna(0.0)
    frame["cash_flow_CFOToOR"] = _to_numeric_series(frame, "ocf_to_or").fillna(0.0)
    frame["dupont_dupontROE"] = roe.fillna(roe_avg).fillna(0.0)

    base = frame[["code", "pubDate", "statDate"]].copy()
    base = base.dropna(subset=["code"]).copy()

    def _table(columns: list[str]) -> pd.DataFrame:
        out = pd.concat([base, frame[columns]], axis=1)
        out = out.drop_duplicates(subset=["code", "pubDate", "statDate"], keep="last")
        return out.reset_index(drop=True)

    tables = {
        "profit": _table(["profit_roeAvg", "profit_npMargin", "profit_gpMargin"]),
        "growth": _table(["growth_YOYNI", "growth_YOYAsset"]),
        "operation": _table(["operation_NRTurnRatio", "operation_INVTurnRatio"]),
        "balance": _table(["balance_currentRatio", "balance_quickRatio"]),
        "cash_flow": _table(["cash_flow_CFOToOR"]),
        "dupont": _table(["dupont_dupontROE"]),
    }
    for key in FINANCIAL_QUERY_METHODS:
        tables.setdefault(key, pd.DataFrame(columns=["code", "pubDate", "statDate"]))
    return tables


def _empty_financial_tables() -> dict[str, pd.DataFrame]:
    return {
        key: pd.DataFrame(columns=["code", "pubDate", "statDate"])
        for key in FINANCIAL_QUERY_METHODS
    }


def _empty_daily_k_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "code",
            "open",
            "high",
            "low",
            "close",
            "preclose",
            "volume",
            "amount",
            "tradestatus",
            "turn",
            "pctChg",
            "peTTM",
            "pbMRQ",
            "psTTM",
            "pcfNcfTTM",
            "isST",
        ]
    )


def _call_tushare(
    callable_obj: Callable[..., Any],
    *,
    strict: bool,
    **kwargs: Any,
) -> pd.DataFrame:
    try:
        frame = callable_obj(**kwargs)
    except Exception:
        if strict:
            raise
        return pd.DataFrame()

    if frame is None:
        return pd.DataFrame()
    if isinstance(frame, pd.DataFrame):
        return frame
    return pd.DataFrame(frame)


def _index_code_candidates(index_code: str) -> tuple[str, ...]:
    normalized = normalize_baostock_code(index_code)
    if normalized in INDEX_DAILY_CODE_CANDIDATES:
        return INDEX_DAILY_CODE_CANDIDATES[normalized]
    return (baostock_to_tushare_code(normalized),)


def _financial_date_range(year_quarters: Iterable[tuple[int, int]]) -> tuple[str, str]:
    normalized = sorted({(int(year), int(quarter)) for year, quarter in year_quarters})
    if not normalized:
        current_year = pd.Timestamp.today().year
        return f"{current_year}0101", f"{current_year}1231"
    min_year = min(year for year, _ in normalized)
    max_year = max(year for year, _ in normalized)
    # Include one extra year for delayed annual report announcements.
    return f"{min_year}0101", f"{max_year + 1}1231"


def _to_tushare_date(value: str) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        digits = "".join(char for char in str(value) if char.isdigit())
        if len(digits) == 8:
            return digits
        raise ValueError(f"Invalid date value: {value!r}")
    return timestamp.strftime("%Y%m%d")


def _to_iso_date(value: Any) -> str:
    if value is None:
        return ""
    raw = str(value).strip()
    if not raw:
        return ""
    timestamp = pd.to_datetime(raw, errors="coerce")
    if pd.isna(timestamp):
        return raw
    return timestamp.strftime("%Y-%m-%d")


def _to_numeric_series(frame: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in frame.columns:
        return pd.Series([pd.NA] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column_name], errors="coerce")


def _maybe_sleep() -> None:
    raw_value = os.getenv("TUSHARE_REQUEST_SLEEP_SECONDS", "0.02").strip()
    if not raw_value:
        return
    try:
        sleep_seconds = float(raw_value)
    except ValueError:
        sleep_seconds = 0.02
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
