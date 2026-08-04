from __future__ import annotations

import sys
import tomllib
from types import SimpleNamespace

import pandas as pd

from alphagraph.data.extend_daily_k import build_akshare_fetchers
from alphagraph.pipeline_config import resolve_config


def test_default_and_ablation_profiles_use_akshare_market_data():
    with open("alphagraph/pipeline_config.toml", "rb") as handle:
        config = tomllib.load(handle)

    profiles = [
        "phase1_relation_aware",
        "graph_report_only_baseline",
        "graph_market_corr",
        "graph_market_corr_industry",
        "graph_hszz_market_corr_industry",
    ]

    for profile in profiles:
        values = resolve_config(config, mode="graph", config_name=profile)
        assert values["MARKET_DATA_SOURCE"] == "akshare", profile


def test_akshare_fetchers_provide_graph_build_interfaces(monkeypatch):
    fake_akshare = SimpleNamespace(
        stock_info_a_code_name=lambda: pd.DataFrame(
            [{"code": "000001", "name": "平安银行"}, {"code": "600000", "name": "浦发银行"}]
        ),
        index_stock_cons=lambda symbol: pd.DataFrame(
            [{"品种代码": "000001", "品种名称": "平安银行"}]
            if symbol == "000300"
            else [{"品种代码": "600000", "品种名称": "浦发银行"}]
        ),
    )
    monkeypatch.setitem(sys.modules, "akshare", fake_akshare)

    fetchers = build_akshare_fetchers()

    basic = fetchers.fetch_stock_basic(["sz.000001"])
    membership = fetchers.fetch_index_membership(["2025-01-31"])
    industry = fetchers.fetch_stock_industry(["2025-01-31"])
    financial = fetchers.fetch_financial_tables(["sz.000001"], [(2025, 1)])

    assert basic[["code", "ipoDate", "type"]].to_dict("records") == [
        {"code": "sz.000001", "ipoDate": "1991-01-01", "type": "1"}
    ]
    assert set(membership["index_name"]) == {"hs300", "zz500"}
    assert set(membership["code"]) == {"sz.000001", "sh.600000"}
    assert {"date", "code", "industry", "industryClassification"}.issubset(industry.columns)
    assert set(financial) == {"profit", "growth", "operation", "balance", "cash_flow", "dupont"}
