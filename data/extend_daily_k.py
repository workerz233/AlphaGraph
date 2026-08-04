from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import socket
import time
from pathlib import Path
from typing import Callable

import pandas as pd

from alphagraph.data.financial_features import FINANCIAL_QUERY_METHODS
from alphagraph.data.market_data import INDEX_CODE_HS300, INDEX_CODE_ZZ500
from alphagraph.data.tushare_fallback import build_tushare_fetchers


FetchDaily = Callable[[list[str], str, str], pd.DataFrame]
FetchBenchmark = Callable[[str, str, str, Path], pd.DataFrame]


def extend_daily_k_artifacts(
    dataset_dir: Path,
    end_date: str,
    start_date: str | None = None,
    source: str = "tushare",
    fetch_daily_k: FetchDaily | None = None,
    fetch_benchmark: FetchBenchmark | None = None,
) -> dict[str, dict[str, object]]:
    dataset_dir = Path(dataset_dir)
    daily_path = dataset_dir / "daily_k.parquet"
    hs300_path = dataset_dir / "hs300_daily_k.parquet"
    zz500_path = dataset_dir / "zz500_daily_k.parquet"

    daily = _read_frame(daily_path)
    codes = sorted(daily["code"].dropna().astype(str).unique()) if "code" in daily.columns else []
    if not codes:
        raise ValueError(f"No stock codes found in {daily_path}")

    fetchers = None
    if fetch_daily_k is None or fetch_benchmark is None:
        fetchers = build_akshare_fetchers() if str(source).strip().lower() == "akshare" else build_tushare_fetchers()
    fetch_daily_k = fetch_daily_k or fetchers.fetch_daily_k  # type: ignore[union-attr]
    fetch_benchmark = fetch_benchmark or fetchers.fetch_benchmark  # type: ignore[union-attr]

    stock_summary = extend_frame_file(
        path=daily_path,
        end_date=end_date,
        start_date=start_date,
        fetch_fn=lambda fetch_start, fetch_end, _cache_path: fetch_daily_k(codes, fetch_start, fetch_end),
        dedupe_columns=["date", "code"],
    )
    hs300_summary = extend_frame_file(
        path=hs300_path,
        end_date=end_date,
        start_date=start_date,
        fetch_fn=lambda fetch_start, fetch_end, cache_path: fetch_benchmark(
            INDEX_CODE_HS300,
            fetch_start,
            fetch_end,
            cache_path,
        ),
        dedupe_columns=["date", "code"],
    )
    zz500_summary = extend_frame_file(
        path=zz500_path,
        end_date=end_date,
        start_date=start_date,
        fetch_fn=lambda fetch_start, fetch_end, cache_path: fetch_benchmark(
            INDEX_CODE_ZZ500,
            fetch_start,
            fetch_end,
            cache_path,
        ),
        dedupe_columns=["date", "code"],
    )
    return {"daily_k": stock_summary, "hs300_daily_k": hs300_summary, "zz500_daily_k": zz500_summary}


def extend_frame_file(
    path: Path,
    end_date: str,
    fetch_fn: Callable[[str, str, Path], pd.DataFrame],
    dedupe_columns: list[str],
    start_date: str | None = None,
) -> dict[str, object]:
    existing = _read_frame(path)
    if existing.empty:
        if start_date is None:
            raise ValueError(f"{path} is empty; start_date is required")
        fetch_start = pd.Timestamp(start_date)
    else:
        existing = _normalize_dates(existing)
        latest = existing["date"].max()
        fetch_start = pd.Timestamp(start_date) if start_date is not None else pd.Timestamp(latest) + pd.Timedelta(days=1)

    fetch_end = pd.Timestamp(end_date)
    if fetch_start > fetch_end:
        return _summary(existing, added_rows=0, fetch_start=fetch_start, fetch_end=fetch_end, path=path)

    cache_path = path.with_name(f"{path.stem}.extend_{fetch_start:%Y%m%d}_{fetch_end:%Y%m%d}.tmp.parquet")
    increment = fetch_fn(fetch_start.strftime("%Y-%m-%d"), fetch_end.strftime("%Y-%m-%d"), cache_path)
    increment = _normalize_dates(increment)
    merged = merge_daily_frames(existing, increment, dedupe_columns=dedupe_columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return _summary(merged, added_rows=len(merged) - len(existing), fetch_start=fetch_start, fetch_end=fetch_end, path=path)


def merge_daily_frames(
    existing: pd.DataFrame,
    increment: pd.DataFrame,
    dedupe_columns: list[str] | None = None,
) -> pd.DataFrame:
    dedupe_columns = dedupe_columns or ["date", "code"]
    frames = [frame for frame in [_normalize_dates(existing), _normalize_dates(increment)] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    sort_columns = [column for column in ["code", "date"] if column in merged.columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns)
    dedupe_existing = [column for column in dedupe_columns if column in merged.columns]
    if dedupe_existing:
        merged = merged.drop_duplicates(subset=dedupe_existing, keep="last")
    return merged.reset_index(drop=True)


def build_akshare_fetchers() -> object:
    akshare_module = importlib.import_module("akshare")
    socket.setdefaulttimeout(AKSHARE_STOCK_TIMEOUT_SECONDS)
    stock_basic_cache: pd.DataFrame | None = None

    def fetch_daily_k(codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for index, code in enumerate(sorted({str(code).strip() for code in codes if str(code).strip()}), start=1):
            symbol = _akshare_symbol(code)
            normalized = _fetch_akshare_stock_daily(
                akshare_module=akshare_module,
                code=code,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
            if not normalized.empty:
                frames.append(normalized)
            if index == 1 or index % 50 == 0:
                print(f"akshare stock progress {index}/{len(codes)}")
            _maybe_sleep()
        if not frames:
            raise RuntimeError(f"AkShare returned empty stock data for {len(codes)} symbols")
        return pd.concat(frames, ignore_index=True)

    def fetch_benchmark(index_code: str, start_date: str, end_date: str, cache_path: Path) -> pd.DataFrame:
        del cache_path
        symbol = _akshare_index_symbol(index_code)
        frame = akshare_module.stock_zh_index_daily(symbol=symbol)
        normalized = _normalize_akshare_index_daily(frame, index_code)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        normalized = normalized.loc[(normalized["date"] >= start) & (normalized["date"] <= end)].copy()
        if normalized.empty:
            raise RuntimeError(f"AkShare returned empty index data for {index_code} in [{start_date}, {end_date}]")
        return normalized

    def fetch_stock_basic(codes: list[str]) -> pd.DataFrame:
        nonlocal stock_basic_cache
        if stock_basic_cache is None:
            stock_basic_cache = _fetch_akshare_stock_basic_all(akshare_module)

        normalized_codes = {_normalize_market_code(code) for code in codes if str(code).strip()}
        basic = stock_basic_cache
        if normalized_codes:
            basic = basic.loc[basic["code"].isin(sorted(normalized_codes))].copy()
        return basic.reset_index(drop=True)

    def fetch_stock_industry(dates: list[str]) -> pd.DataFrame:
        nonlocal stock_basic_cache
        if stock_basic_cache is None:
            stock_basic_cache = _fetch_akshare_stock_basic_all(akshare_module)
        unique_dates = sorted({_iso_date(value) for value in dates if str(value).strip()})
        if not unique_dates or stock_basic_cache.empty:
            return pd.DataFrame(columns=["date", "code", "industry", "industryClassification"])

        industry_base = stock_basic_cache[["code", "industry", "industryClassification"]].copy()
        industry_base = industry_base.drop_duplicates(subset=["code"], keep="last")
        date_frame = pd.DataFrame({"date": unique_dates})
        date_frame["_key"] = 1
        industry_base["_key"] = 1
        merged = date_frame.merge(industry_base, on="_key", how="inner").drop(columns=["_key"])
        return merged[["date", "code", "industry", "industryClassification"]]

    def fetch_index_membership(dates: list[str]) -> pd.DataFrame:
        unique_dates = sorted({_iso_date(value) for value in dates if str(value).strip()})
        if not unique_dates:
            return pd.DataFrame(columns=["date", "code", "index_name"])

        rows: list[dict[str, object]] = []
        for index_name, symbol in {"hs300": "000300", "zz500": "000905"}.items():
            frame = _fetch_akshare_index_cons(akshare_module, symbol=symbol)
            if frame.empty:
                continue
            for date_text in unique_dates:
                for code in frame["code"].dropna().astype(str):
                    rows.append(
                        {
                            "date": date_text,
                            "code": _normalize_market_code(code),
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
        del codes, year_quarters
        base_columns = ["code", "pubDate", "statDate"]
        return {
            table_name: pd.DataFrame(columns=base_columns)
            for table_name in FINANCIAL_QUERY_METHODS
        }

    class _Fetchers:
        pass

    fetchers = _Fetchers()
    fetchers.fetch_daily_k = fetch_daily_k
    fetchers.fetch_benchmark = fetch_benchmark
    fetchers.fetch_stock_basic = fetch_stock_basic
    fetchers.fetch_stock_industry = fetch_stock_industry
    fetchers.fetch_index_membership = fetch_index_membership
    fetchers.fetch_financial_tables = fetch_financial_tables
    return fetchers


def _fetch_akshare_stock_basic_all(akshare_module: object) -> pd.DataFrame:
    try:
        frame = akshare_module.stock_info_a_code_name()
    except Exception:
        frame = pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code", "ipoDate", "type", "industry", "industryClassification"])

    basic = frame.copy()
    code_column = _first_existing_column(basic, ["code", "代码", "证券代码", "股票代码"])
    name_column = _first_existing_column(basic, ["name", "名称", "股票简称"])
    if not code_column:
        return pd.DataFrame(columns=["code", "ipoDate", "type", "industry", "industryClassification"])

    basic["code"] = basic[code_column].astype(str).map(_normalize_market_code)
    basic["ipoDate"] = "1991-01-01"
    basic["type"] = "1"
    basic["industry"] = basic[name_column].astype(str) if name_column else ""
    basic["industryClassification"] = "akshare_stock_basic"
    return (
        basic[["code", "ipoDate", "type", "industry", "industryClassification"]]
        .drop_duplicates(subset=["code"], keep="last")
        .reset_index(drop=True)
    )


def _fetch_akshare_index_cons(akshare_module: object, symbol: str) -> pd.DataFrame:
    fetch_attempts = [
        ("index_stock_cons", {"symbol": symbol}),
        ("index_stock_cons_csindex", {"symbol": symbol}),
    ]
    for method_name, kwargs in fetch_attempts:
        method = getattr(akshare_module, method_name, None)
        if method is None:
            continue
        try:
            frame = method(**kwargs)
        except TypeError:
            try:
                frame = method(symbol)
            except Exception:
                continue
        except Exception:
            continue
        normalized = _normalize_akshare_index_cons(frame)
        if not normalized.empty:
            return normalized
    return pd.DataFrame(columns=["code"])


def _normalize_akshare_index_cons(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["code"])
    cons = frame.copy()
    code_column = _first_existing_column(
        cons,
        ["品种代码", "成分券代码", "证券代码", "股票代码", "code", "con_code"],
    )
    if not code_column:
        return pd.DataFrame(columns=["code"])
    return pd.DataFrame(
        {"code": cons[code_column].dropna().astype(str).map(_normalize_market_code)}
    ).drop_duplicates().reset_index(drop=True)


def _first_existing_column(frame: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    return ""


AKSHARE_STOCK_TIMEOUT_SECONDS = 30


def _call_akshare_with_timeout(fn, *args, timeout: int = AKSHARE_STOCK_TIMEOUT_SECONDS, **kwargs):
    """在独立线程中调用 akshare 函数，超时则取消并抛出 TimeoutError。"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise TimeoutError(f"akshare call timed out after {timeout}s") from None


def _fetch_akshare_stock_daily(
    akshare_module: object,
    code: str,
    symbol: str,
    start_date: str,
    end_date: str,
    retries: int = 3,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            frame = _call_akshare_with_timeout(
                akshare_module.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=_compact_date(start_date),
                end_date=_compact_date(end_date),
                adjust="qfq",
            )
            normalized = _normalize_akshare_stock_daily(frame, code)
            if not normalized.empty:
                return normalized
        except Exception as exc:
            last_error = exc
        try:
            frame = _call_akshare_with_timeout(
                akshare_module.stock_zh_a_daily,
                symbol=_akshare_prefixed_symbol(code),
                start_date=_compact_date(start_date),
                end_date=_compact_date(end_date),
                adjust="qfq",
            )
            normalized = _normalize_akshare_daily_stock_daily(frame, code)
            if not normalized.empty:
                return normalized
        except Exception as exc:
            last_error = exc
        time.sleep(0.5 * attempt)
    print(f"akshare stock fetch failed for {code}: {last_error}")
    return _empty_daily_frame()


def _normalize_akshare_stock_daily(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_daily_frame()
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily.get("日期"), errors="coerce")
    daily["code"] = _normalize_market_code(code)
    daily["open"] = pd.to_numeric(daily.get("开盘"), errors="coerce")
    daily["high"] = pd.to_numeric(daily.get("最高"), errors="coerce")
    daily["low"] = pd.to_numeric(daily.get("最低"), errors="coerce")
    daily["close"] = pd.to_numeric(daily.get("收盘"), errors="coerce")
    daily["volume"] = pd.to_numeric(daily.get("成交量"), errors="coerce")
    daily["amount"] = pd.to_numeric(daily.get("成交额"), errors="coerce")
    daily["turn"] = pd.to_numeric(daily.get("换手率"), errors="coerce").fillna(0.0)
    daily["pctChg"] = pd.to_numeric(daily.get("涨跌幅"), errors="coerce")
    return _finalize_daily_frame(daily)


def _normalize_akshare_daily_stock_daily(frame: pd.DataFrame, code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_daily_frame()
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily.get("date"), errors="coerce")
    daily["code"] = _normalize_market_code(code)
    daily["open"] = pd.to_numeric(daily.get("open"), errors="coerce")
    daily["high"] = pd.to_numeric(daily.get("high"), errors="coerce")
    daily["low"] = pd.to_numeric(daily.get("low"), errors="coerce")
    daily["close"] = pd.to_numeric(daily.get("close"), errors="coerce")
    daily["volume"] = pd.to_numeric(daily.get("volume"), errors="coerce")
    daily["amount"] = pd.to_numeric(daily.get("amount"), errors="coerce")
    daily["turn"] = pd.to_numeric(daily.get("turnover"), errors="coerce").fillna(0.0) * 100.0
    daily["pctChg"] = pd.NA
    return _finalize_daily_frame(daily)


def _normalize_akshare_index_daily(frame: pd.DataFrame, index_code: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_daily_frame()
    daily = frame.copy()
    daily["date"] = pd.to_datetime(daily.get("date"), errors="coerce")
    daily["code"] = _normalize_market_code(index_code)
    daily["open"] = pd.to_numeric(daily.get("open"), errors="coerce")
    daily["high"] = pd.to_numeric(daily.get("high"), errors="coerce")
    daily["low"] = pd.to_numeric(daily.get("low"), errors="coerce")
    daily["close"] = pd.to_numeric(daily.get("close"), errors="coerce")
    daily["volume"] = pd.to_numeric(daily.get("volume"), errors="coerce")
    daily["amount"] = 0.0
    daily["turn"] = 0.0
    daily["pctChg"] = pd.NA
    return _finalize_daily_frame(daily)


def _finalize_daily_frame(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.dropna(subset=["date", "code", "open", "high", "low", "close"]).copy()
    if daily.empty:
        return _empty_daily_frame()
    daily = daily.sort_values(["code", "date"], ignore_index=True)
    daily["preclose"] = daily.groupby("code")["close"].shift(1)
    pct_from_close = (daily["close"] / daily["preclose"] - 1.0) * 100.0
    daily["pctChg"] = pd.to_numeric(daily["pctChg"], errors="coerce").fillna(pct_from_close)
    daily["tradestatus"] = 1
    daily["adjustflag"] = "2"
    daily["peTTM"] = 0.0
    daily["pbMRQ"] = 0.0
    daily["psTTM"] = 0.0
    daily["pcfNcfTTM"] = 0.0
    daily["isST"] = 0
    return daily[
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
            "adjustflag",
            "turn",
            "tradestatus",
            "pctChg",
            "peTTM",
            "pbMRQ",
            "psTTM",
            "pcfNcfTTM",
            "isST",
        ]
    ].reset_index(drop=True)


def _empty_daily_frame() -> pd.DataFrame:
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
            "adjustflag",
            "turn",
            "tradestatus",
            "pctChg",
            "peTTM",
            "pbMRQ",
            "psTTM",
            "pcfNcfTTM",
            "isST",
        ]
    )


def _akshare_symbol(code: str) -> str:
    normalized = _normalize_market_code(code)
    return normalized.split(".", 1)[1]


def _akshare_index_symbol(index_code: str) -> str:
    normalized = _normalize_market_code(index_code)
    market, symbol = normalized.split(".", 1)
    return f"{market}{symbol}"


def _akshare_prefixed_symbol(code: str) -> str:
    normalized = _normalize_market_code(code)
    market, symbol = normalized.split(".", 1)
    return f"{market}{symbol}"


def _normalize_market_code(code: str) -> str:
    raw = str(code).strip().lower()
    if raw.startswith(("sh.", "sz.")):
        return raw
    symbol = raw.split(".")[0]
    if symbol.startswith(("6", "9")):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


def _compact_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _iso_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _maybe_sleep() -> None:
    time.sleep(0.05)


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.read_pickle(path)


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    normalized = frame.copy()
    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        normalized = normalized.dropna(subset=["date"])
    return normalized


def _summary(
    frame: pd.DataFrame,
    added_rows: int,
    fetch_start: pd.Timestamp,
    fetch_end: pd.Timestamp,
    path: Path,
) -> dict[str, object]:
    if frame.empty or "date" not in frame.columns:
        min_date = None
        max_date = None
    else:
        min_date = pd.Timestamp(frame["date"].min()).strftime("%Y-%m-%d")
        max_date = pd.Timestamp(frame["date"].max()).strftime("%Y-%m-%d")
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "added_rows": int(added_rows),
        "min_date": min_date,
        "max_date": max_date,
        "fetch_start": fetch_start.strftime("%Y-%m-%d"),
        "fetch_end": fetch_end.strftime("%Y-%m-%d"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend alphagraph daily K artifacts")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--source", choices=("tushare", "akshare"), default="tushare")
    args = parser.parse_args()

    summary = extend_daily_k_artifacts(
        dataset_dir=args.dataset_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        source=args.source,
    )
    for name, values in summary.items():
        print(
            f"{name}: rows={values['rows']} added={values['added_rows']} "
            f"range={values['min_date']}..{values['max_date']}"
        )


if __name__ == "__main__":
    main()
