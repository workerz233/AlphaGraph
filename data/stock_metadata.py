from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from alphagraph.data.market_data import normalize_baostock_code


INDEX_QUERY_NAMES = {
    "hs300": "query_hs300_stocks",
    "zz500": "query_zz500_stocks",
    "sz50": "query_sz50_stocks",
}


def result_to_dataframe(result: Any) -> pd.DataFrame:
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock query failed: {result.error_code} {result.error_msg}")

    rows: list[list[Any]] = []
    while result.next():
        rows.append(result.get_row_data())
    return pd.DataFrame(rows, columns=getattr(result, "fields", []))


def fetch_stock_basic_for_codes(
    codes: Iterable[str],
    bs_client: Any,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for code in sorted({normalize_baostock_code(code) for code in codes}):
        frame = result_to_dataframe(bs_client.query_stock_basic(code=code))
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["code", "ipoDate", "type"])

    basic = pd.concat(frames, ignore_index=True)
    if "code" in basic.columns:
        basic["code"] = basic["code"].map(normalize_baostock_code)
    return basic.drop_duplicates(subset=["code"], keep="last").reset_index(drop=True)


def fetch_stock_industry_for_dates(
    dates: Iterable[str],
    bs_client: Any,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date in sorted({str(date) for date in dates if str(date)}):
        frame = result_to_dataframe(bs_client.query_stock_industry(code="", date=date))
        if not frame.empty:
            frame["date"] = date
            if "code" in frame.columns:
                frame["code"] = frame["code"].map(normalize_baostock_code)
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=["date", "code", "industry", "industryClassification"]
        )
    return pd.concat(frames, ignore_index=True)


def fetch_index_membership_for_dates(
    dates: Iterable[str],
    bs_client: Any,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for date in sorted({str(date) for date in dates if str(date)}):
        for index_name, query_name in INDEX_QUERY_NAMES.items():
            query_fn = getattr(bs_client, query_name)
            frame = result_to_dataframe(query_fn(date=date))
            if frame.empty:
                continue
            frame["date"] = date
            frame["index_name"] = index_name
            if "code" in frame.columns:
                frame["code"] = frame["code"].map(normalize_baostock_code)
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["date", "code", "index_name"])
    return pd.concat(frames, ignore_index=True)


def build_monthly_stock_metadata_features(
    monthly_features_df: pd.DataFrame,
    stock_basic_df: pd.DataFrame,
    industry_df: pd.DataFrame,
    index_membership_df: pd.DataFrame,
) -> pd.DataFrame:
    monthly = monthly_features_df.copy()
    if monthly.empty:
        return monthly

    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["code"] = monthly["code"].map(normalize_baostock_code)

    basic = _normalize_basic_frame(stock_basic_df)
    monthly = monthly.merge(basic, on="code", how="left")
    monthly["listing_age_days"] = (
        monthly["date"] - pd.to_datetime(monthly["ipoDate"], errors="coerce")
    ).dt.days
    monthly["listing_age_days"] = monthly["listing_age_days"].clip(lower=0).fillna(0).astype(float)
    monthly["security_type_id"] = pd.to_numeric(monthly["type"], errors="coerce").fillna(0).astype(float)
    monthly = monthly.drop(columns=[column for column in ["ipoDate", "type"] if column in monthly.columns])

    industry = _normalize_industry_frame(industry_df)
    monthly = monthly.merge(
        industry,
        on=["date", "code"],
        how="left",
    )
    monthly["industry"] = monthly["industry"].fillna("")
    monthly["industryClassification"] = monthly["industryClassification"].fillna("")
    industry_keys = sorted(key for key in monthly["industry"].unique().tolist() if key)
    industry_lookup = {industry: index + 1 for index, industry in enumerate(industry_keys)}
    monthly["industry_id"] = monthly["industry"].map(industry_lookup).fillna(0).astype(int)

    membership = _normalize_membership_frame(index_membership_df)
    for index_name in INDEX_QUERY_NAMES:
        monthly[f"is_{index_name}"] = 0.0
    if not membership.empty:
        membership["present"] = 1.0
        pivot = (
            membership.pivot_table(
                index=["date", "code"],
                columns="index_name",
                values="present",
                aggfunc="max",
                fill_value=0.0,
            )
            .reset_index()
            .rename_axis(columns=None)
        )
        monthly = monthly.merge(pivot, on=["date", "code"], how="left")
        for index_name in INDEX_QUERY_NAMES:
            raw_column = index_name
            target_column = f"is_{index_name}"
            if raw_column in monthly.columns:
                monthly[target_column] = monthly[raw_column].fillna(0.0).astype(float)
                monthly = monthly.drop(columns=[raw_column])

    for index_name in INDEX_QUERY_NAMES:
        monthly[f"is_{index_name}"] = monthly[f"is_{index_name}"].fillna(0.0).astype(float)

    return monthly.drop(
        columns=[column for column in ["status", "outDate"] if column in monthly.columns]
    )


def _normalize_basic_frame(stock_basic_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["code", "ipoDate", "type"]
    if stock_basic_df.empty:
        return pd.DataFrame(columns=columns)
    basic = stock_basic_df.copy()
    basic["code"] = basic["code"].map(normalize_baostock_code)
    return basic.reindex(columns=columns).drop_duplicates(subset=["code"], keep="last")


def _normalize_industry_frame(industry_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "code", "industry", "industryClassification"]
    if industry_df.empty:
        return pd.DataFrame(columns=columns)
    industry = industry_df.copy()
    industry["date"] = pd.to_datetime(industry["date"])
    industry["code"] = industry["code"].map(normalize_baostock_code)
    return industry.reindex(columns=columns).drop_duplicates(
        subset=["date", "code"],
        keep="last",
    )


def _normalize_membership_frame(index_membership_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "code", "index_name"]
    if index_membership_df.empty:
        return pd.DataFrame(columns=columns)
    membership = index_membership_df.copy()
    membership["date"] = pd.to_datetime(membership["date"])
    membership["code"] = membership["code"].map(normalize_baostock_code)
    return membership.reindex(columns=columns).drop_duplicates()
