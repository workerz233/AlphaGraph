from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from alphagraph.data.stock_metadata import result_to_dataframe
from alphagraph.data.market_data import normalize_baostock_code


FINANCIAL_QUERY_METHODS = {
    "profit": "query_profit_data",
    "growth": "query_growth_data",
    "operation": "query_operation_data",
    "balance": "query_balance_data",
    "cash_flow": "query_cash_flow_data",
    "dupont": "query_dupont_data",
}
FINANCIAL_BASE_COLUMNS = ["code", "pubDate", "statDate"]
FINANCIAL_FEATURE_COLUMNS = [
    "profit_roeAvg",
    "profit_npMargin",
    "profit_gpMargin",
    "growth_YOYNI",
    "growth_YOYAsset",
    "operation_NRTurnRatio",
    "operation_INVTurnRatio",
    "balance_currentRatio",
    "balance_quickRatio",
    "cash_flow_CFOToOR",
    "dupont_dupontROE",
]


def fetch_quarterly_financial_tables(
    codes: Iterable[str],
    year_quarters: Iterable[tuple[int, int]],
    bs_client: Any,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, list[pd.DataFrame]] = {
        table_name: [] for table_name in FINANCIAL_QUERY_METHODS
    }
    normalized_codes = sorted({normalize_baostock_code(code) for code in codes})

    for code in normalized_codes:
        for year, quarter in year_quarters:
            for table_name, method_name in FINANCIAL_QUERY_METHODS.items():
                result = getattr(bs_client, method_name)(
                    code=code,
                    year=int(year),
                    quarter=int(quarter),
                )
                frame = _prefix_financial_frame(result_to_dataframe(result), table_name)
                if not frame.empty:
                    tables[table_name].append(frame)

    return {
        table_name: (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=FINANCIAL_BASE_COLUMNS)
        )
        for table_name, frames in tables.items()
    }


def build_monthly_financial_features(
    monthly_features_df: pd.DataFrame,
    financial_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    monthly = monthly_features_df.copy()
    if monthly.empty:
        return monthly

    monthly["date"] = pd.to_datetime(monthly["date"])
    monthly["code"] = monthly["code"].map(normalize_baostock_code)
    financial = _combine_financial_tables(financial_tables)

    output_rows: list[dict[str, Any]] = []
    for row in monthly.to_dict(orient="records"):
        code = row["code"]
        month_end_date = row["date"]
        visible = financial.loc[
            (financial["code"] == code)
            & financial["pubDate"].notna()
            & (financial["pubDate"] <= month_end_date)
        ]
        enriched = dict(row)
        if visible.empty:
            enriched.update({column: 0.0 for column in FINANCIAL_FEATURE_COLUMNS})
            enriched["financial_report_age_days"] = 0.0
            enriched["has_financial_features"] = 0.0
        else:
            latest = visible.sort_values(["pubDate", "statDate"]).iloc[-1]
            for column in FINANCIAL_FEATURE_COLUMNS:
                enriched[column] = float(latest.get(column, 0.0) or 0.0)
            enriched["financial_report_age_days"] = float(
                (month_end_date - latest["pubDate"]).days
            )
            enriched["has_financial_features"] = 1.0
        output_rows.append(enriched)

    return pd.DataFrame(output_rows)


def _prefix_financial_frame(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=FINANCIAL_BASE_COLUMNS)

    normalized = frame.copy()
    normalized["code"] = normalized["code"].map(normalize_baostock_code)
    rename_map = {
        column: f"{table_name}_{column}"
        for column in normalized.columns
        if column not in FINANCIAL_BASE_COLUMNS
    }
    normalized = normalized.rename(columns=rename_map)
    for column in normalized.columns:
        if column not in FINANCIAL_BASE_COLUMNS:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized


def _combine_financial_tables(financial_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined: pd.DataFrame | None = None
    for table in financial_tables.values():
        if table.empty:
            continue
        frame = table.copy()
        frame["code"] = frame["code"].map(normalize_baostock_code)
        frame["pubDate"] = pd.to_datetime(frame["pubDate"], errors="coerce")
        frame["statDate"] = pd.to_datetime(frame["statDate"], errors="coerce")
        combined = (
            frame
            if combined is None
            else combined.merge(frame, on=FINANCIAL_BASE_COLUMNS, how="outer")
        )

    if combined is None:
        combined = pd.DataFrame(columns=FINANCIAL_BASE_COLUMNS)

    for column in FINANCIAL_FEATURE_COLUMNS:
        if column not in combined.columns:
            combined[column] = 0.0
        combined[column] = pd.to_numeric(combined[column], errors="coerce").fillna(0.0)

    combined["pubDate"] = pd.to_datetime(combined["pubDate"], errors="coerce")
    combined["statDate"] = pd.to_datetime(combined["statDate"], errors="coerce")
    return combined
