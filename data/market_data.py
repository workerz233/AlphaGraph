from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import baostock as bs
except ModuleNotFoundError:  # pragma: no cover
    bs = None


DAILY_K_FIELDS = (
    "date,code,open,high,low,close,preclose,volume,amount,"
    "adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"
)
INDEX_DAILY_K_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"
DAILY_K_BASE_COLUMNS = [
    "date",
    "code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "tradestatus",
    "turn",
]
DAILY_K_OPTIONAL_COLUMNS = [
    "preclose",
    "adjustflag",
    "pctChg",
    "peTTM",
    "pbMRQ",
    "psTTM",
    "pcfNcfTTM",
    "isST",
]
ARTIFACT_DIR = (
    Path(__file__).resolve().parents[2] / "artifacts" / "alphagraph"
)
INDEX_CODE_HS300 = "sh.000300"
INDEX_CODE_ZZ500 = "sh.000905"
FORWARD_RETURN_HORIZONS = (1, 3, 6)


def normalize_baostock_code(code: str) -> str:
    normalized = code.strip().lower()
    if normalized.startswith(("sh.", "sz.")):
        return normalized

    base = normalized.split(".")[0]
    if base.startswith(("6", "9")):
        return f"sh.{base}"
    return f"sz.{base}"


def fetch_daily_k_for_codes(
    codes: list[str] | set[str],
    start_date: str,
    end_date: str,
    adjustflag: str = "2",
    bs_client: Any | None = None,
    fields: str = DAILY_K_FIELDS,
) -> pd.DataFrame:
    client = bs_client or bs
    if client is None:
        raise RuntimeError("baostock is required to fetch daily K data")

    rows: list[dict[str, Any]] = []
    for code in sorted({normalize_baostock_code(code) for code in codes}):
        result = client.query_history_k_data_plus(
            code,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag,
        )
        if result.error_code != "0":
            raise RuntimeError(
                f"query_history_k_data_plus failed for {code}: {result.error_code} {result.error_msg}"
            )

        while result.next():
            row = result.get_row_data()
            result_fields = getattr(result, "fields", None) or fields.split(",")
            rows.append(dict(zip(result_fields, row, strict=True)))

    daily_df = pd.DataFrame(rows)
    return _normalize_daily_table(daily_df)


def build_monthly_stock_features(
    daily_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    stock_daily = _add_rolling_features(_normalize_daily_table(daily_df))
    benchmark_daily = _add_benchmark_features(_normalize_daily_table(benchmark_df))

    stock_monthly = _last_rows_by_month(stock_daily)
    benchmark_view = benchmark_daily[
        ["date", "benchmark_ret_20d", "benchmark_ret_60d"]
    ].drop_duplicates(subset=["date"])
    monthly = stock_monthly.merge(benchmark_view, on="date", how="left")
    monthly["rel_ret_20d_hs300"] = (
        monthly["ret_20d"] - monthly["benchmark_ret_20d"].fillna(0.0)
    )
    monthly["rel_ret_60d_hs300"] = (
        monthly["ret_60d"] - monthly["benchmark_ret_60d"].fillna(0.0)
    )
    monthly["is_tradable"] = (
        (monthly["tradestatus"] == 1) & (monthly["volume"] > 0) & (monthly["close"] > 0)
    )
    monthly["pe_ttm"] = monthly["peTTM"].fillna(0.0)
    monthly["pb_mrq"] = monthly["pbMRQ"].fillna(0.0)
    monthly["ps_ttm"] = monthly["psTTM"].fillna(0.0)
    monthly["pcf_ncf_ttm"] = monthly["pcfNcfTTM"].fillna(0.0)
    monthly["is_st"] = monthly["isST"].fillna(0.0)
    return monthly[
        [
            "month",
            "date",
            "code",
            "month_end_close",
            "ret_20d",
            "ret_60d",
            "rsj_20d",
            "rsj_60d",
            "vol_20d",
            "vol_60d",
            "amount_20d_mean",
            "turn_20d_mean",
            "rel_ret_20d_hs300",
            "rel_ret_60d_hs300",
            "pct_chg_20d_mean",
            "pct_chg_20d_std",
            "pe_ttm",
            "pb_mrq",
            "ps_ttm",
            "pcf_ncf_ttm",
            "is_st",
            "is_tradable",
        ]
    ].sort_values(["month", "code"], ignore_index=True)


def build_future_labels(
    daily_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    required_horizons: tuple[int, ...] = FORWARD_RETURN_HORIZONS,
) -> pd.DataFrame:
    invalid_horizons = set(required_horizons) - set(FORWARD_RETURN_HORIZONS)
    if invalid_horizons:
        raise ValueError(
            f"required_horizons must be a subset of {FORWARD_RETURN_HORIZONS}: "
            f"{sorted(invalid_horizons)}"
        )
    normalized_daily = _normalize_daily_table(daily_df)
    normalized_benchmark = _normalize_daily_table(benchmark_df)
    stock_monthly = _last_rows_by_month(normalized_daily)
    benchmark_monthly = _last_rows_by_month(normalized_benchmark)[
        ["month", "month_end_close"]
    ].rename(columns={"month_end_close": "benchmark_month_end_close"})
    for horizon in FORWARD_RETURN_HORIZONS:
        benchmark_monthly[f"future_benchmark_return_{horizon}m"] = (
            benchmark_monthly["benchmark_month_end_close"].shift(-horizon)
            / benchmark_monthly["benchmark_month_end_close"]
            - 1.0
        )

    stock_monthly = stock_monthly.sort_values(["code", "month"], ignore_index=True)
    grouped_close = stock_monthly.groupby("code")["month_end_close"]
    for horizon in FORWARD_RETURN_HORIZONS:
        stock_monthly[f"future_stock_return_{horizon}m"] = (
            grouped_close.shift(-horizon) / stock_monthly["month_end_close"] - 1.0
        )
    stock_monthly["next_month_is_tradable"] = stock_monthly.groupby("code")[
        "tradestatus"
    ].shift(-1)

    labels = stock_monthly.merge(
        benchmark_monthly[
            ["month"]
            + [f"future_benchmark_return_{horizon}m" for horizon in FORWARD_RETURN_HORIZONS]
        ],
        on="month",
        how="left",
    )
    for horizon in FORWARD_RETURN_HORIZONS:
        labels[f"future_excess_return_{horizon}m"] = (
            labels[f"future_stock_return_{horizon}m"]
            - labels[f"future_benchmark_return_{horizon}m"]
        )
    labels = labels.merge(
        _build_forward_sharpe_labels(
            daily_df=normalized_daily,
            benchmark_df=normalized_benchmark,
            stock_monthly=stock_monthly,
        ),
        on=["month", "code"],
        how="left",
    )
    required_return_columns = [
        column
        for horizon in required_horizons
        for column in (
            f"future_stock_return_{horizon}m",
            f"future_benchmark_return_{horizon}m",
        )
    ]
    labels = labels.dropna(subset=required_return_columns)
    labels = labels[
        [
            "month",
            "code",
            *[
                column
                for horizon in FORWARD_RETURN_HORIZONS
                for column in (
                    f"future_stock_return_{horizon}m",
                    f"future_benchmark_return_{horizon}m",
                    f"future_excess_return_{horizon}m",
                    f"future_stock_sharpe_{horizon}m",
                    f"future_excess_sharpe_{horizon}m",
                )
            ],
            "next_month_is_tradable",
        ]
    ].sort_values(["month", "code"], ignore_index=True)
    labels = labels.rename(columns={"next_month_is_tradable": "is_tradable"})
    labels["is_tradable"] = labels["is_tradable"].fillna(0).astype(int).astype(bool)
    return labels


def _build_forward_sharpe_labels(
    daily_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    stock_monthly: pd.DataFrame,
) -> pd.DataFrame:
    daily_returns = _daily_return_frame(daily_df, "stock_daily_return")
    benchmark_returns = _daily_return_frame(benchmark_df, "benchmark_daily_return")[
        ["date", "benchmark_daily_return"]
    ].drop_duplicates(subset=["date"])
    daily_returns = daily_returns.merge(benchmark_returns, on="date", how="left")
    daily_returns["excess_daily_return"] = (
        daily_returns["stock_daily_return"]
        - daily_returns["benchmark_daily_return"].fillna(0.0)
    )

    rows: list[dict[str, Any]] = []
    monthly_view = stock_monthly[["month", "code", "date"]].sort_values(["code", "month"])
    for row in monthly_view.itertuples(index=False):
        record: dict[str, Any] = {"month": row.month, "code": row.code}
        month_end = pd.Timestamp(row.date)
        for horizon in FORWARD_RETURN_HORIZONS:
            horizon_end = month_end + pd.offsets.MonthEnd(horizon)
            window = daily_returns.loc[
                (daily_returns["code"] == row.code)
                & (daily_returns["date"] > month_end)
                & (daily_returns["date"] <= horizon_end)
            ]
            record[f"future_stock_sharpe_{horizon}m"] = _annualized_sharpe(
                window["stock_daily_return"]
            )
            record[f"future_excess_sharpe_{horizon}m"] = _annualized_sharpe(
                window["excess_daily_return"]
            )
        rows.append(record)
    return pd.DataFrame(rows)


def _daily_return_frame(daily_df: pd.DataFrame, return_column: str) -> pd.DataFrame:
    normalized = daily_df.sort_values(["code", "date"]).copy()
    normalized[return_column] = normalized.groupby("code")["close"].pct_change()
    return normalized[["date", "code", return_column]]


def _annualized_sharpe(returns: pd.Series) -> float:
    cleaned = pd.to_numeric(returns, errors="coerce").dropna()
    if cleaned.shape[0] < 2:
        return 0.0
    std = float(cleaned.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(cleaned.mean() / std * (252 ** 0.5))


def load_or_fetch_hs300_benchmark(
    start_date: str,
    end_date: str,
    cache_path: Path | None = None,
    bs_client: Any | None = None,
) -> pd.DataFrame:
    target_path = cache_path or ARTIFACT_DIR / "hs300_daily_k.parquet"
    return load_or_fetch_index_benchmark(
        index_code=INDEX_CODE_HS300,
        start_date=start_date,
        end_date=end_date,
        cache_path=target_path,
        bs_client=bs_client,
    )


def load_or_fetch_index_benchmark(
    index_code: str,
    start_date: str,
    end_date: str,
    cache_path: Path,
    bs_client: Any | None = None,
) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    benchmark_df = fetch_daily_k_for_codes(
        [index_code],
        start_date=start_date,
        end_date=end_date,
        adjustflag="2",
        bs_client=bs_client,
        fields=INDEX_DAILY_K_FIELDS,
    )
    _write_parquet(benchmark_df, cache_path)
    return benchmark_df


def build_monthly_index_returns(
    daily_df: pd.DataFrame,
    target_months: set[str],
    return_column: str,
    horizon_months: int = 1,
) -> pd.DataFrame:
    monthly = _last_rows_by_month(_normalize_daily_table(daily_df))[["month", "month_end_close"]].copy()
    monthly = monthly.sort_values("month", ignore_index=True)
    monthly["future_month_close"] = monthly["month_end_close"].shift(-int(horizon_months))
    monthly[return_column] = monthly["future_month_close"] / monthly["month_end_close"] - 1.0
    monthly = monthly.loc[
        monthly["month"].isin(sorted(target_months)),
        ["month", return_column],
    ].dropna(subset=[return_column])
    return monthly.drop_duplicates(subset=["month"]).sort_values("month", ignore_index=True)


def _normalize_daily_table(daily_df: pd.DataFrame) -> pd.DataFrame:
    if daily_df.empty:
        return pd.DataFrame(
            columns=DAILY_K_BASE_COLUMNS + DAILY_K_OPTIONAL_COLUMNS
        )

    normalized = daily_df.copy()
    for column in DAILY_K_BASE_COLUMNS + DAILY_K_OPTIONAL_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = pd.NA
    normalized["date"] = pd.to_datetime(normalized["date"])
    normalized["code"] = normalized["code"].map(normalize_baostock_code)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "preclose",
        "volume",
        "amount",
        "turn",
        "pctChg",
        "peTTM",
        "pbMRQ",
        "psTTM",
        "pcfNcfTTM",
        "isST",
    ]
    for column in numeric_columns:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["tradestatus"] = (
        pd.to_numeric(normalized["tradestatus"], errors="coerce").fillna(0).astype(int)
    )
    normalized["isST"] = pd.to_numeric(normalized["isST"], errors="coerce").fillna(0).astype(int)
    return normalized.sort_values(["code", "date"], ignore_index=True)


def _add_rolling_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    enriched = daily_df.copy()
    enriched["daily_return"] = enriched.groupby("code")["close"].pct_change().fillna(0.0)

    for window in (20, 60):
        enriched[f"ret_{window}d"] = enriched.groupby("code")["close"].transform(
            lambda values: _window_return(values, window)
        )
        enriched[f"vol_{window}d"] = enriched.groupby("code")["daily_return"].transform(
            lambda values: values.rolling(window=window, min_periods=1).std().fillna(0.0)
        )
        enriched[f"rsj_{window}d"] = enriched.groupby("code")["daily_return"].transform(
            lambda values: _rolling_rsj(values, window)
        )

    enriched["amount_20d_mean"] = enriched.groupby("code")["amount"].transform(
        lambda values: values.rolling(window=20, min_periods=1).mean()
    )
    enriched["turn_20d_mean"] = enriched.groupby("code")["turn"].transform(
        lambda values: values.rolling(window=20, min_periods=1).mean()
    )
    pct_change = pd.to_numeric(enriched["pctChg"], errors="coerce") / 100.0
    enriched["pct_chg"] = pct_change.fillna(enriched["daily_return"])
    enriched["pct_chg_20d_mean"] = enriched.groupby("code")["pct_chg"].transform(
        lambda values: values.rolling(window=20, min_periods=1).mean()
    )
    enriched["pct_chg_20d_std"] = enriched.groupby("code")["pct_chg"].transform(
        lambda values: values.rolling(window=20, min_periods=1).std().fillna(0.0)
    )
    return enriched


def _add_benchmark_features(benchmark_df: pd.DataFrame) -> pd.DataFrame:
    enriched = _add_rolling_features(benchmark_df)
    return enriched.rename(
        columns={
            "ret_20d": "benchmark_ret_20d",
            "ret_60d": "benchmark_ret_60d",
        }
    )


def _last_rows_by_month(daily_df: pd.DataFrame) -> pd.DataFrame:
    monthly = daily_df.copy()
    monthly["month"] = monthly["date"].dt.to_period("M").astype(str)
    monthly = (
        monthly.sort_values(["code", "date"])
        .groupby(["code", "month"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    monthly["month_end_close"] = monthly["close"]
    return monthly


def _window_return(values: pd.Series, window: int) -> pd.Series:
    anchor = values.shift(window - 1)
    if not values.empty:
        anchor = anchor.fillna(values.iloc[0])
    return values / anchor - 1.0


def _rolling_rsj(values: pd.Series, window: int) -> pd.Series:
    returns = pd.to_numeric(values, errors="coerce").fillna(0.0)
    upside = returns.where(returns > 0.0, 0.0).pow(2)
    downside = returns.where(returns < 0.0, 0.0).pow(2)
    total = returns.pow(2).rolling(window=window, min_periods=1).sum()
    skew = (
        upside.rolling(window=window, min_periods=1).sum()
        - downside.rolling(window=window, min_periods=1).sum()
    )
    return (skew / total.replace(0.0, pd.NA)).fillna(0.0)


def _write_parquet(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
