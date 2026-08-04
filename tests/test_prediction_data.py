from __future__ import annotations

import pandas as pd
import pytest

from alphagraph.data.market_data import build_future_labels
from alphagraph.graph import build


def _monthly_daily_frame(code: str, closes: list[float]) -> pd.DataFrame:
    months = pd.period_range("2026-01", periods=len(closes), freq="M")
    return pd.DataFrame(
        [
            {
                "date": month.end_time.normalize(),
                "code": code,
                "close": close,
                "tradestatus": 1,
            }
            for month, close in zip(months, closes, strict=True)
        ]
    )


def test_explicit_target_months_form_a_contiguous_calendar():
    assert build.derive_target_months("202601", "202607") == {
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
        "2026-07",
    }


def test_prediction_fetch_range_stops_at_as_of_date():
    assert build.derive_fetch_date_range(
        ["2026-01", "2026-07"],
        as_of_date="2026-07-31",
    ) == ("2025-10-03", "2026-07-31")


def test_prediction_market_cache_requires_exact_cutoff_date():
    incomplete = pd.DataFrame(
        {
            "date": ["2025-10-03", "2026-07-01"],
            "code": ["sz.000001", "sz.000001"],
        }
    )

    assert not build._market_frame_covers_date_range(
        incomplete,
        required_start_date="2025-10-03",
        required_end_date="2026-07-31",
    )


def test_prediction_dated_cache_requires_exact_snapshot_dates():
    incomplete = pd.DataFrame(
        {
            "date": ["2026-06-30", "2026-07-01"],
            "code": ["sz.000001", "sz.000001"],
        }
    )

    assert not build._dated_frame_covers_months(
        incomplete,
        required_dates=["2026-06-30", "2026-07-31"],
    )


def test_future_labels_can_require_only_one_month_return():
    daily = _monthly_daily_frame("sz.000001", [100.0, 110.0])
    benchmark = _monthly_daily_frame("sh.000300", [100.0, 105.0])

    labels = build_future_labels(daily, benchmark, required_horizons=(1,))

    assert labels[["month", "code"]].to_dict(orient="records") == [
        {"month": "2026-01", "code": "sz.000001"}
    ]
    assert labels.iloc[0]["future_stock_return_1m"] == pytest.approx(0.1)
    assert pd.isna(labels.iloc[0]["future_stock_return_3m"])
    assert pd.isna(labels.iloc[0]["future_stock_return_6m"])


def test_future_labels_still_require_all_horizons_by_default():
    daily = _monthly_daily_frame("sz.000001", [100.0, 110.0])
    benchmark = _monthly_daily_frame("sh.000300", [100.0, 105.0])

    assert build_future_labels(daily, benchmark).empty


def test_prediction_cli_forwards_calendar_and_as_of_arguments(monkeypatch, tmp_path):
    captured: dict[str, object] = {}

    def fake_prepare_monthly_dataset(**kwargs):
        captured.update(kwargs)
        return {"artifact_dir": tmp_path}

    monkeypatch.setattr(build, "prepare_monthly_dataset", fake_prepare_monthly_dataset)

    build.main(
        [
            "--workspace-root",
            str(tmp_path),
            "--markdown-dir",
            "reportdata",
            "--data-start-month",
            "202401",
            "--signal-month",
            "202607",
            "--as-of-date",
            "2026-07-31",
            "--prediction",
        ]
    )

    assert captured["data_start_month"] == "202401"
    assert captured["signal_month"] == "202607"
    assert captured["as_of_date"] == "2026-07-31"
    assert captured["prediction_mode"] is True


def test_market_data_is_clipped_to_as_of_date_even_when_cache_has_future_rows():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-31", "2026-08-03"]),
            "code": ["sz.000001", "sz.000001"],
            "close": [10.0, 11.0],
        }
    )

    clipped = build.clip_market_frame_as_of(frame, "2026-07-31")

    assert clipped["date"].tolist() == [pd.Timestamp("2026-07-31")]


def test_prediction_dated_metadata_is_clipped_before_building_stock_universe():
    frame = pd.DataFrame(
        {
            "date": ["2026-07-31", "2026-08-03"],
            "code": ["sz.000001", "sz.000002"],
            "index_name": ["hs300", "hs300"],
        }
    )

    clipped = build.clip_dated_frame_as_of(frame, "2026-07-31")

    assert clipped[["date", "code"]].to_dict(orient="records") == [
        {"date": "2026-07-31", "code": "sz.000001"}
    ]


def test_prediction_financial_queries_and_rows_stop_at_as_of_date():
    quarters = build._derive_financial_year_quarters(
        ["2024-01", "2026-07"],
        as_of_date="2026-07-31",
    )
    assert (2026, 2) in quarters
    assert (2026, 3) not in quarters

    tables = {
        "profit": pd.DataFrame(
            {
                "code": ["sz.000001", "sz.000001"],
                "pubDate": ["2026-07-30", "2026-08-01"],
                "statDate": ["2026-06-30", "2026-06-30"],
            }
        )
    }
    clipped = build.clip_financial_tables_as_of(tables, "2026-07-31")
    assert clipped["profit"]["pubDate"].tolist() == ["2026-07-30"]


def test_prediction_benchmark_returns_can_require_only_one_month():
    hs300 = _monthly_daily_frame("sh.000300", [100.0, 105.0])
    zz500 = _monthly_daily_frame("sh.000905", [100.0, 104.0])

    result = build.build_monthly_dual_benchmark_returns(
        hs300,
        zz500,
        target_months={"2026-01"},
        required_horizons=(1,),
    )

    assert result["month"].tolist() == ["2026-01"]
    assert result.iloc[0]["hs300_return_1m"] == pytest.approx(0.05)
    assert pd.isna(result.iloc[0]["hs300_return_3m"])


def test_prediction_reports_are_clipped_to_calendar_and_as_of_date():
    reports = pd.DataFrame(
        [
            {"report_month": "2023-12", "publish_date": "2023-12-31"},
            {"report_month": "2026-07", "publish_date": "2026-07-15"},
            {"report_month": "2026-07", "publish_date": "2026-08-01"},
        ]
    )

    result = build.clip_reports_for_prediction(
        reports,
        data_start_month="202401",
        signal_month="202607",
        as_of_date="2026-07-31",
    )

    assert result.to_dict(orient="records") == [
        {"report_month": "2026-07", "publish_date": "2026-07-15"}
    ]


def test_prediction_metadata_dates_do_not_extend_past_as_of_date():
    assert build._derive_target_month_end_dates(
        ["2026-06", "2026-07"],
        as_of_date="2026-07-15",
    ) == ["2026-06-30", "2026-07-15"]
