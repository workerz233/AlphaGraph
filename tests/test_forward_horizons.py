from __future__ import annotations

import pandas as pd
import pytest

from alphagraph.backtest.runner import run_monthly_backtest
from alphagraph.data.market_data import build_future_labels
from alphagraph.graph.build import (
    _load_cached_or_fetch_dated_frame,
    _load_cached_or_fetch_market_frame,
    derive_fetch_date_range,
)


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


def test_fetch_date_range_extends_six_months_after_last_signal_month():
    assert derive_fetch_date_range(["2026-01"]) == ("2025-10-03", "2026-07-31")


def test_future_labels_include_one_three_and_six_month_returns():
    daily = _monthly_daily_frame("sz.000001", [100, 110, 121, 133.1, 146.41, 161.051, 177.1561])
    benchmark = _monthly_daily_frame("sh.000300", [100, 105, 110.25, 115.7625, 121.550625, 127.628156, 134.009564])

    labels = build_future_labels(daily, benchmark)
    january = labels.loc[(labels["month"] == "2026-01") & (labels["code"] == "sz.000001")].iloc[0]

    assert january["future_stock_return_1m"] == pytest.approx(0.10)
    assert january["future_stock_return_3m"] == pytest.approx(0.331)
    assert january["future_stock_return_6m"] == pytest.approx(0.771561)
    assert january["future_benchmark_return_6m"] == pytest.approx(0.34009564)
    assert january["future_excess_return_6m"] == pytest.approx(0.43146536)


def test_future_labels_include_forward_excess_sharpe():
    dates = pd.to_datetime(
        [
            "2026-01-31",
            "2026-02-01",
            "2026-02-02",
            "2026-02-28",
            "2026-03-31",
            "2026-04-30",
            "2026-05-31",
            "2026-06-30",
            "2026-07-31",
        ]
    )
    daily = pd.DataFrame(
        {
            "date": dates,
            "code": "sz.000001",
            "close": [100.0, 102.0, 103.02, 106.1106, 107.171706, 108.243423, 109.325857, 110.419116, 111.523307],
            "tradestatus": 1,
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "code": "sh.000300",
            "close": [100.0, 101.0, 101.505, 102.52005, 103.545251, 104.580704, 105.626511, 106.682776, 107.749604],
            "tradestatus": 1,
        }
    )

    labels = build_future_labels(daily, benchmark)
    january = labels.loc[(labels["month"] == "2026-01") & (labels["code"] == "sz.000001")].iloc[0]

    stock_returns = pd.Series([0.02, 0.01, 0.03])
    benchmark_returns = pd.Series([0.01, 0.005, 0.01])
    excess_returns = stock_returns - benchmark_returns
    expected_sharpe = excess_returns.mean() / excess_returns.std(ddof=1) * (252 ** 0.5)

    assert january["future_stock_sharpe_1m"] == pytest.approx(
        stock_returns.mean() / stock_returns.std(ddof=1) * (252 ** 0.5)
    )
    assert january["future_excess_sharpe_1m"] == pytest.approx(expected_sharpe)


def test_monthly_backtest_outputs_three_and_six_month_forward_returns():
    scores = pd.DataFrame([{"month": "2026-01", "code": "sz.000001", "score": 1.0}])
    labels = pd.DataFrame(
        [
            {
                "month": "2026-01",
                "code": "sz.000001",
                "future_stock_return_1m": 0.10,
                "future_stock_return_3m": 0.331,
                "future_stock_return_6m": 0.771561,
                "future_excess_return_1m": 0.05,
                "future_excess_return_3m": 0.181,
                "future_excess_return_6m": 0.431561,
                "is_tradable": True,
            }
        ]
    )
    benchmark = pd.DataFrame(
        [
            {
                "month": "2026-01",
                "benchmark_return_1m": 0.05,
                "hs300_return_1m": 0.05,
                "zz500_return_1m": 0.04,
                "benchmark_return_3m": 0.15,
                "hs300_return_3m": 0.15,
                "zz500_return_3m": 0.12,
                "benchmark_return_6m": 0.34,
                "hs300_return_6m": 0.34,
                "zz500_return_6m": 0.30,
            }
        ]
    )

    result = run_monthly_backtest(scores, labels, benchmark, cost_bps=0, k=1)

    row = result.iloc[0]
    assert row["portfolio_return"] == pytest.approx(0.10)
    assert row["portfolio_return_3m"] == pytest.approx(0.331)
    assert row["portfolio_return_6m"] == pytest.approx(0.771561)
    assert row["excess_hs300_return_6m"] == pytest.approx(0.431561)


def test_market_cache_is_refetched_when_it_does_not_cover_required_end_date(tmp_path):
    cache_path = tmp_path / "daily_k.parquet"
    _monthly_daily_frame("sz.000001", [100, 110]).to_parquet(cache_path)
    fetched = _monthly_daily_frame("sz.000001", [100, 110, 121, 133.1, 146.41, 161.051, 177.1561])

    result = _load_cached_or_fetch_market_frame(
        cache_path,
        lambda: fetched,
        required_end_date="2026-07-31",
    )

    assert pd.to_datetime(result["date"]).max() == pd.Timestamp("2026-07-31")


def test_market_cache_is_refetched_when_it_does_not_cover_required_start_date(tmp_path):
    cache_path = tmp_path / "daily_k.parquet"
    cached = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-07-31"]),
            "code": "sz.000001",
            "close": [100.0, 110.0],
        }
    )
    cached.to_parquet(cache_path)
    fetched = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-10-03", "2026-07-31"]),
            "code": "sz.000001",
            "close": [90.0, 110.0],
        }
    )

    result = _load_cached_or_fetch_market_frame(
        cache_path,
        lambda: fetched,
        required_start_date="2025-10-03",
        required_end_date="2026-07-31",
    )

    assert pd.to_datetime(result["date"]).min() == pd.Timestamp("2025-10-03")


def test_dated_metadata_cache_is_refetched_when_required_months_are_missing(tmp_path):
    cache_path = tmp_path / "index_membership.parquet"
    pd.DataFrame(
        [{"date": "2026-01-31", "code": "sz.000001", "index_name": "hs300"}]
    ).to_parquet(cache_path)
    fetched = pd.DataFrame(
        [
            {"date": "2025-03-31", "code": "sz.000001", "index_name": "hs300"},
            {"date": "2025-04-30", "code": "sz.000001", "index_name": "hs300"},
        ]
    )

    result = _load_cached_or_fetch_dated_frame(
        cache_path,
        lambda: fetched,
        required_dates=["2025-03-31", "2025-04-30"],
    )

    assert set(pd.to_datetime(result["date"]).dt.to_period("M").astype(str)) == {
        "2025-03",
        "2025-04",
    }
