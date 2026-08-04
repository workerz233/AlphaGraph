from __future__ import annotations

from typing import Any

import pandas as pd


TRADING_STRATEGIES = ("topk", "topk_buffer", "momentum_confirmed", "cluster_topk")


def apply_stock_filters(
    scored_df: pd.DataFrame,
    monthly_features_df: pd.DataFrame | None,
    liquidity_quantile: float = 0.0,
    min_price: float = 0.0,
    exclude_st: bool = True,
) -> pd.DataFrame:
    if monthly_features_df is None or monthly_features_df.empty:
        return scored_df.copy()

    feature_columns = [
        column
        for column in [
            "month",
            "code",
            "is_tradable",
            "is_st",
            "month_end_close",
            "amount_20d_mean",
            "ret_20d",
            "ret_60d",
            "rsj_20d",
            "rsj_60d",
        ]
        if column in monthly_features_df.columns
    ]
    filtered = scored_df.merge(
        monthly_features_df[feature_columns].drop_duplicates(subset=["month", "code"]),
        on=["month", "code"],
        how="left",
    )

    if "amount_20d_mean" in filtered.columns and float(liquidity_quantile) > 0.0:
        amount = pd.to_numeric(filtered["amount_20d_mean"], errors="coerce")
        filtered = filtered.assign(_amount_20d_mean=amount)
        thresholds = filtered.groupby("month")["_amount_20d_mean"].transform(
            lambda values: values.quantile(float(liquidity_quantile))
        )
        filtered = filtered.loc[filtered["_amount_20d_mean"].fillna(0.0) >= thresholds.fillna(float("-inf"))]
        filtered = filtered.drop(columns=["_amount_20d_mean"])
    if "is_tradable" in filtered.columns:
        filtered = filtered.loc[filtered["is_tradable"].fillna(False)].copy()
    if exclude_st and "is_st" in filtered.columns:
        filtered = filtered.loc[pd.to_numeric(filtered["is_st"], errors="coerce").fillna(0.0) == 0.0].copy()
    if "month_end_close" in filtered.columns:
        close = pd.to_numeric(filtered["month_end_close"], errors="coerce")
        filtered = filtered.loc[close.fillna(0.0) >= float(min_price)].copy()

    return filtered.reset_index(drop=True)


def build_topk_portfolio(scores_df: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    ranked = _rank_scores(scores_df)
    return ranked.loc[ranked["rank"] <= int(k)].reset_index(drop=True)


def build_topk_buffer_portfolio(
    scores_df: pd.DataFrame,
    monthly_features_df: pd.DataFrame | None,
    previous_codes: set[str],
    buy_topk: int = 20,
    hold_topk: int = 50,
    max_positions: int = 20,
    liquidity_quantile: float = 0.2,
    min_price: float = 2.0,
    exclude_st: bool = True,
    require_positive_ret_20d: bool = False,
    require_positive_ret_60d: bool = False,
) -> pd.DataFrame:
    filtered = apply_stock_filters(
        scores_df,
        monthly_features_df,
        liquidity_quantile=liquidity_quantile,
        min_price=min_price,
        exclude_st=exclude_st,
    )
    ranked = _rank_scores(filtered)
    if ranked.empty:
        return ranked

    hold_codes = ranked.loc[
        (ranked["rank"] <= int(hold_topk)) & ranked["code"].isin(previous_codes),
        "code",
    ].tolist()
    selected_codes = list(dict.fromkeys(hold_codes))

    buy_candidates = ranked.loc[ranked["rank"] <= int(buy_topk), "code"].tolist()
    for code in buy_candidates:
        if code not in selected_codes:
            selected_codes.append(code)
        if len(selected_codes) >= int(max_positions):
            break

    if len(selected_codes) < int(max_positions):
        hold_candidates = ranked.loc[ranked["rank"] <= int(hold_topk), "code"].tolist()
        for code in hold_candidates:
            if code not in selected_codes:
                selected_codes.append(code)
            if len(selected_codes) >= int(max_positions):
                break

    selected = ranked.loc[ranked["code"].isin(selected_codes)].copy()
    selected["_selection_order"] = selected["code"].map({code: index for index, code in enumerate(selected_codes)})
    return selected.sort_values("_selection_order").drop(columns=["_selection_order"]).reset_index(drop=True)


def build_momentum_confirmed_portfolio(
    scores_df: pd.DataFrame,
    monthly_features_df: pd.DataFrame | None,
    candidate_topk: int = 40,
    final_topk: int = 10,
    require_positive_ret_20d: bool = True,
    require_positive_ret_60d: bool = True,
    liquidity_quantile: float = 0.3,
    min_price: float = 2.0,
    exclude_st: bool = True,
) -> pd.DataFrame:
    filtered = apply_stock_filters(
        scores_df,
        monthly_features_df,
        liquidity_quantile=liquidity_quantile,
        min_price=min_price,
        exclude_st=exclude_st,
    )
    ranked = _rank_scores(filtered)
    candidates = ranked.loc[ranked["rank"] <= int(candidate_topk)].copy()
    if require_positive_ret_20d and "ret_20d" in candidates.columns:
        candidates = candidates.loc[pd.to_numeric(candidates["ret_20d"], errors="coerce").fillna(0.0) > 0.0]
    if require_positive_ret_60d and "ret_60d" in candidates.columns:
        candidates = candidates.loc[pd.to_numeric(candidates["ret_60d"], errors="coerce").fillna(0.0) > 0.0]
    return candidates.sort_values(["score", "code"], ascending=[False, True]).head(int(final_topk)).reset_index(drop=True)


def build_cluster_topk_portfolio(
    scores_df: pd.DataFrame,
    monthly_features_df: pd.DataFrame | None,
    stock_clusters_df: pd.DataFrame | None,
    candidate_topk: int = 80,
    max_positions: int = 20,
    max_per_cluster: int = 3,
    liquidity_quantile: float = 0.2,
    min_price: float = 2.0,
    exclude_st: bool = True,
    require_positive_ret_20d: bool = False,
    require_positive_ret_60d: bool = False,
    require_positive_rsj_20d: bool = False,
    require_positive_rsj_60d: bool = False,
) -> pd.DataFrame:
    filtered = apply_stock_filters(
        scores_df,
        monthly_features_df,
        liquidity_quantile=liquidity_quantile,
        min_price=min_price,
        exclude_st=exclude_st,
    )
    if filtered.empty:
        return filtered
    require_rsj_20d = bool(require_positive_rsj_20d or require_positive_ret_20d)
    require_rsj_60d = bool(require_positive_rsj_60d or require_positive_ret_60d)
    if require_rsj_20d:
        filtered = _filter_positive_signal(filtered, "rsj_20d")
    if require_rsj_60d:
        filtered = _filter_positive_signal(filtered, "rsj_60d")
    if filtered.empty:
        return filtered
    enriched = filtered.copy()
    if stock_clusters_df is not None and not stock_clusters_df.empty:
        cluster_columns = [
            column
            for column in [
                "month",
                "code",
                "cluster_id",
                "cluster_score",
                "pagerank",
                "report_attention",
            ]
            if column in stock_clusters_df.columns
        ]
        enriched = enriched.merge(
            stock_clusters_df[cluster_columns].drop_duplicates(subset=["month", "code"]),
            on=["month", "code"],
            how="left",
        )
    for column in ("cluster_score", "pagerank", "report_attention"):
        if column not in enriched.columns:
            enriched[column] = 0.0
        enriched[column] = pd.to_numeric(enriched[column], errors="coerce").fillna(0.0)
    enriched["cluster_id"] = enriched.get("cluster_id", -1)
    enriched["cluster_rank_score"] = (
        0.65 * _zscore_by_month(enriched, "score")
        + 0.20 * _zscore_by_month(enriched, "cluster_score")
        + 0.10 * _zscore_by_month(enriched, "pagerank")
        + 0.05 * _zscore_by_month(enriched, "report_attention")
    )
    ranked = enriched.sort_values(
        ["month", "cluster_rank_score", "score", "code"],
        ascending=[True, False, False, True],
    ).copy()
    ranked["rank"] = ranked.groupby("month").cumcount() + 1
    ranked = ranked.loc[ranked["rank"] <= int(candidate_topk)].copy()

    selected_frames: list[pd.DataFrame] = []
    for _month, month_rows in ranked.groupby("month", sort=True):
        cluster_counts: dict[object, int] = {}
        selected_indices: list[int] = []
        for index, row in month_rows.iterrows():
            cluster_id = row.get("cluster_id", -1)
            used = cluster_counts.get(cluster_id, 0)
            if used >= int(max_per_cluster):
                continue
            selected_indices.append(index)
            cluster_counts[cluster_id] = used + 1
            if len(selected_indices) >= int(max_positions):
                break
        selected_frames.append(month_rows.loc[selected_indices].copy())
    if not selected_frames:
        return ranked.iloc[0:0].copy()
    return pd.concat(selected_frames, ignore_index=True)


def build_portfolio_by_strategy(
    trading_strategy: str,
    scores_df: pd.DataFrame,
    monthly_features_df: pd.DataFrame | None,
    previous_codes: set[str],
    k: int = 20,
    buy_topk: int = 20,
    hold_topk: int = 50,
    max_positions: int = 20,
    liquidity_quantile: float = 0.2,
    min_price: float = 2.0,
    exclude_st: bool = True,
    momentum_candidate_topk: int = 40,
    momentum_final_topk: int = 10,
    require_positive_ret_20d: bool = True,
    require_positive_ret_60d: bool = True,
    momentum_liquidity_quantile: float = 0.3,
    stock_clusters_df: pd.DataFrame | None = None,
    cluster_candidate_topk: int = 80,
    cluster_max_per_cluster: int = 3,
    cluster_require_positive_ret_20d: bool = False,
    cluster_require_positive_ret_60d: bool = False,
    cluster_require_positive_rsj_20d: bool = False,
    cluster_require_positive_rsj_60d: bool = False,
) -> pd.DataFrame:
    strategy = str(trading_strategy).strip().lower()
    if strategy == "topk":
        filtered = apply_stock_filters(
            scores_df,
            monthly_features_df,
            liquidity_quantile=0.0,
            min_price=0.0,
            exclude_st=exclude_st,
        )
        return build_topk_portfolio(filtered, k=k)
    if strategy == "topk_buffer":
        return build_topk_buffer_portfolio(
            scores_df,
            monthly_features_df,
            previous_codes=previous_codes,
            buy_topk=buy_topk,
            hold_topk=hold_topk,
            max_positions=max_positions,
            liquidity_quantile=liquidity_quantile,
            min_price=min_price,
            exclude_st=exclude_st,
        )
    if strategy == "momentum_confirmed":
        return build_momentum_confirmed_portfolio(
            scores_df,
            monthly_features_df,
            candidate_topk=momentum_candidate_topk,
            final_topk=momentum_final_topk,
            require_positive_ret_20d=require_positive_ret_20d,
            require_positive_ret_60d=require_positive_ret_60d,
            liquidity_quantile=momentum_liquidity_quantile,
            min_price=min_price,
            exclude_st=exclude_st,
        )
    if strategy == "cluster_topk":
        return build_cluster_topk_portfolio(
            scores_df=scores_df,
            monthly_features_df=monthly_features_df,
            stock_clusters_df=stock_clusters_df,
            candidate_topk=cluster_candidate_topk,
            max_positions=max_positions,
            max_per_cluster=cluster_max_per_cluster,
            liquidity_quantile=liquidity_quantile,
            min_price=min_price,
            exclude_st=exclude_st,
            require_positive_ret_20d=cluster_require_positive_ret_20d,
            require_positive_ret_60d=cluster_require_positive_ret_60d,
            require_positive_rsj_20d=cluster_require_positive_rsj_20d,
            require_positive_rsj_60d=cluster_require_positive_rsj_60d,
        )
    raise ValueError(f"Unsupported trading_strategy={trading_strategy!r}; expected one of {TRADING_STRATEGIES}")


def _rank_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    ranked = scores_df.sort_values(["month", "score", "code"], ascending=[True, False, True]).copy()
    ranked["rank"] = ranked.groupby("month").cumcount() + 1
    return ranked


def _filter_positive_signal(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in frame.columns:
        return frame.iloc[0:0].copy()
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame.loc[values > 0.0].copy()


def _zscore_by_month(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.groupby(frame["month"]).transform(_zscore_series).fillna(0.0)


def _zscore_series(values: pd.Series) -> pd.Series:
    values = values.fillna(values.median())
    std = values.std(ddof=0)
    if not std:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


def run_daily_signal_backtest(
    scores_df: pd.DataFrame,
    daily_k_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    monthly_features_df: pd.DataFrame | None = None,
    trading_strategy: str = "topk",
    cost_bps: float = 20.0,
    k: int = 20,
    buy_topk: int = 20,
    hold_topk: int = 50,
    max_positions: int = 20,
    liquidity_quantile: float = 0.2,
    min_price: float = 2.0,
    exclude_st: bool = True,
    momentum_candidate_topk: int = 40,
    momentum_final_topk: int = 10,
    require_positive_ret_20d: bool = True,
    require_positive_ret_60d: bool = True,
    momentum_liquidity_quantile: float = 0.3,
    stock_clusters_df: pd.DataFrame | None = None,
    cluster_candidate_topk: int = 80,
    cluster_max_per_cluster: int = 3,
    cluster_require_positive_ret_20d: bool = False,
    cluster_require_positive_ret_60d: bool = False,
    cluster_require_positive_rsj_20d: bool = False,
    cluster_require_positive_rsj_60d: bool = False,
    entry_delay_days: int = 1,
    stop_loss_pct: float = 0.08,
    trailing_stop_pct: float = 0.0,
    limit_threshold_pct: float = 9.8,
    slippage_bps: float = 5.0,
    max_monthly_turnover: float = 1.0,
    single_stock_max_weight: float = 0.10,
    max_hold_months: int = 0,
    entry_batch_days: int = 1,
    stop_loss_confirm_days: int = 1,
    entry_trigger: str = "immediate",
    entry_window_days: int = 15,
    min_pullback_pct: float = 0.03,
    max_pullback_pct: float = 0.08,
    breakout_lookback_days: int = 20,
    breakout_volume_multiplier: float = 1.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored = scores_df.dropna(subset=["month", "code", "score"]).copy()
    daily = _prepare_daily_k(daily_k_df)
    if scored.empty or daily.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    calendar = sorted(daily["date"].dropna().unique())
    previous_target_codes: set[str] = set()
    holdings: dict[str, dict[str, float | pd.Timestamp]] = {}
    daily_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    cumulative_net_value = 1.0

    months = sorted(scored["month"].astype(str).unique())
    for index, month in enumerate(months):
        next_month = months[index + 1] if index + 1 < len(months) else None
        month_scored = scored.loc[scored["month"].astype(str) == month].copy()
        month_features = None
        if monthly_features_df is not None and not monthly_features_df.empty:
            month_features = monthly_features_df.loc[monthly_features_df["month"].astype(str) == month].copy()
        target_rows = build_portfolio_by_strategy(
            trading_strategy=trading_strategy,
            scores_df=month_scored,
            monthly_features_df=month_features,
            previous_codes=previous_target_codes,
            k=k,
            buy_topk=buy_topk,
            hold_topk=hold_topk,
            max_positions=max_positions,
            liquidity_quantile=liquidity_quantile,
            min_price=min_price,
            exclude_st=exclude_st,
            momentum_candidate_topk=momentum_candidate_topk,
            momentum_final_topk=momentum_final_topk,
            require_positive_ret_20d=require_positive_ret_20d,
            require_positive_ret_60d=require_positive_ret_60d,
            momentum_liquidity_quantile=momentum_liquidity_quantile,
            stock_clusters_df=(
                stock_clusters_df.loc[stock_clusters_df["month"].astype(str) == str(month)].copy()
                if stock_clusters_df is not None and not stock_clusters_df.empty
                else None
            ),
            cluster_candidate_topk=cluster_candidate_topk,
            cluster_max_per_cluster=cluster_max_per_cluster,
            cluster_require_positive_ret_20d=cluster_require_positive_ret_20d,
            cluster_require_positive_ret_60d=cluster_require_positive_ret_60d,
            cluster_require_positive_rsj_20d=cluster_require_positive_rsj_20d,
            cluster_require_positive_rsj_60d=cluster_require_positive_rsj_60d,
        )
        target_codes = target_rows["code"].astype(str).tolist()[: int(max_positions)]
        target_code_set = set(target_codes)
        buy_weight = _target_buy_weight(target_codes, single_stock_max_weight)
        previous_target_codes = target_code_set

        start_date = _signal_start_date(month, month_features, calendar, int(entry_delay_days))
        end_date = _signal_start_date(next_month, None, calendar, int(entry_delay_days)) if next_month else None
        period_dates = [
            date
            for date in calendar
            if start_date is not None and date >= start_date and (end_date is None or date < end_date)
        ]
        if not period_dates:
            monthly_rows.append(_empty_month_row(month, benchmark_df, cumulative_net_value))
            continue

        starting_codes = set(holdings)
        month_buy_events: list[str] = []
        month_sell_events: list[str] = []
        month_turnover = 0.0
        month_cost = 0.0
        max_buys = max(0, int(round(max(float(max_monthly_turnover), 0.0) * max(int(max_positions), 1))))
        sold_for_rebalance: set[str] = set()
        blocked_after_risk_exit: set[str] = set()
        observed_highs: dict[str, float] = {}

        for day_index, date in enumerate(period_dates):
            day = daily.loc[daily["date"] == date].set_index("code", drop=False)
            for code in sorted(list(holdings)):
                if code not in target_code_set and code not in sold_for_rebalance:
                    if code in day.index and _can_sell(day.loc[code], limit_threshold_pct):
                        position_weight = float(holdings.get(code, {}).get("weight", 0.0))
                        _append_trade_event(event_rows, month, date, code, "sell", "rebalance", day.loc[code], slippage_bps, position_weight)
                        month_sell_events.append(_format_event(date, code, "rebalance"))
                        holdings.pop(code, None)
                        sold_for_rebalance.add(code)
                        month_turnover += position_weight
                        month_cost += (float(cost_bps) + float(slippage_bps)) / 10000.0 * position_weight

            for code in sorted(list(holdings)):
                if code not in day.index:
                    continue
                row = day.loc[code]
                close = float(row["close"])
                position = holdings[code]
                position["highest_close"] = max(float(position.get("highest_close", close)), close)
                entry_price = float(position.get("entry_price", close))
                stop_hit = float(stop_loss_pct) > 0.0 and close <= entry_price * (1.0 - float(stop_loss_pct))
                position["stop_loss_days"] = int(position.get("stop_loss_days", 0)) + 1 if stop_hit else 0
                stop_confirmed = stop_hit and int(position["stop_loss_days"]) >= max(int(stop_loss_confirm_days), 1)
                trailing_hit = (
                    float(trailing_stop_pct) > 0.0
                    and close <= float(position["highest_close"]) * (1.0 - float(trailing_stop_pct))
                )
                max_hold_hit = _max_hold_hit(position, date, max_hold_months)
                if (stop_confirmed or trailing_hit or max_hold_hit) and _can_sell(row, limit_threshold_pct):
                    reason = "stop_loss" if stop_confirmed else "trailing_stop" if trailing_hit else "max_hold"
                    position_weight = float(position.get("weight", 0.0))
                    _append_trade_event(event_rows, month, date, code, "sell", reason, row, slippage_bps, position_weight)
                    month_sell_events.append(_format_event(date, code, reason))
                    holdings.pop(code, None)
                    blocked_after_risk_exit.add(code)
                    month_turnover += position_weight
                    month_cost += (float(cost_bps) + float(slippage_bps)) / 10000.0 * position_weight

            buys_this_month = len([event for event in month_buy_events if event])
            for code in target_codes:
                if code in blocked_after_risk_exit:
                    continue
                if not _within_entry_batch(target_codes, code, day_index, entry_batch_days):
                    continue
                if code in holdings or buys_this_month >= max_buys or len(holdings) >= int(max_positions):
                    continue
                if code not in day.index or not _can_buy(day.loc[code], exclude_st, limit_threshold_pct):
                    continue
                entry_reason = _entry_reason(
                    daily=daily,
                    day=day,
                    code=code,
                    date=date,
                    day_index=day_index,
                    observed_highs=observed_highs,
                    entry_trigger=entry_trigger,
                    entry_window_days=entry_window_days,
                    min_pullback_pct=min_pullback_pct,
                    max_pullback_pct=max_pullback_pct,
                    breakout_lookback_days=breakout_lookback_days,
                    breakout_volume_multiplier=breakout_volume_multiplier,
                )
                if entry_reason is None:
                    continue
                buy_price = _trade_price(day.loc[code], "buy", slippage_bps)
                holdings[code] = {
                    "entry_price": buy_price,
                    "highest_close": float(day.loc[code]["close"]),
                    "entry_date": pd.Timestamp(date),
                    "stop_loss_days": 0,
                    "weight": buy_weight,
                }
                _append_trade_event(event_rows, month, date, code, "buy", entry_reason, day.loc[code], slippage_bps, buy_weight)
                month_buy_events.append(_format_event(date, code, entry_reason))
                buys_this_month += 1
                month_turnover += buy_weight
                month_cost += (float(cost_bps) + float(slippage_bps)) / 10000.0 * buy_weight

            day_return = _held_daily_return(holdings, day)
            if day_return != 0.0:
                cumulative_net_value *= 1.0 + day_return
            if month_cost > 0.0:
                cumulative_net_value *= 1.0 - month_cost
                day_return -= month_cost
                month_cost = 0.0
            daily_rows.append(
                {
                    "date": pd.Timestamp(date),
                    "month": month,
                    "portfolio_return": day_return,
                    "holding_count": int(len(holdings)),
                    "cumulative_net_value": cumulative_net_value,
                    "holding_codes": ",".join(sorted(holdings)),
                }
            )

        current_codes = set(holdings)
        benchmark_hs300_return, benchmark_zz500_return = _period_benchmark_returns(
            daily=daily,
            period_dates=period_dates,
            fallback_benchmark_df=benchmark_df,
            month=month,
        )
        first_net_value = daily_rows[-len(period_dates) - 1]["cumulative_net_value"] if len(daily_rows) > len(period_dates) else 1.0
        portfolio_return = cumulative_net_value / first_net_value - 1.0
        monthly_rows.append(
            {
                "month": month,
                "selected_count": int(len(current_codes)),
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_hs300_return,
                "benchmark_hs300_return": benchmark_hs300_return,
                "benchmark_zz500_return": benchmark_zz500_return,
                "excess_return": portfolio_return - benchmark_hs300_return,
                "excess_hs300_return": portfolio_return - benchmark_hs300_return,
                "excess_zz500_return": portfolio_return - benchmark_zz500_return,
                "turnover": min(month_turnover, 1.0),
                "transaction_cost": 0.0,
                "ic": _compute_month_ic(month_scored),
                "cumulative_net_value": cumulative_net_value,
                "buy_codes": ",".join(sorted({event.split(":")[1] for event in month_buy_events})),
                "sell_codes": ",".join(sorted({event.split(":")[1] for event in month_sell_events})),
                "hold_codes": ",".join(sorted(current_codes & starting_codes)),
                "buy_events": ";".join(month_buy_events),
                "sell_events": ";".join(month_sell_events),
            }
        )

    return pd.DataFrame(monthly_rows), pd.DataFrame(daily_rows), pd.DataFrame(event_rows)


def _prepare_daily_k(daily_k_df: pd.DataFrame) -> pd.DataFrame:
    if daily_k_df is None or daily_k_df.empty:
        return pd.DataFrame()
    required = {"date", "code", "open", "close"}
    if not required.issubset(daily_k_df.columns):
        return pd.DataFrame()
    daily = daily_k_df.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["code"] = daily["code"].astype(str)
    for column in ["open", "high", "low", "close", "preclose", "pctChg", "tradestatus", "isST"]:
        if column in daily.columns:
            daily[column] = pd.to_numeric(daily[column], errors="coerce")
    if "high" not in daily.columns:
        daily["high"] = daily[["open", "close"]].max(axis=1)
    if "low" not in daily.columns:
        daily["low"] = daily[["open", "close"]].min(axis=1)
    if "preclose" not in daily.columns:
        daily["preclose"] = daily.groupby("code")["close"].shift(1)
    if "pctChg" not in daily.columns:
        daily["pctChg"] = (daily["close"] / daily["preclose"] - 1.0) * 100.0
    if "tradestatus" not in daily.columns:
        daily["tradestatus"] = 1
    if "isST" not in daily.columns:
        daily["isST"] = 0
    return daily.dropna(subset=["date", "code", "open", "close"]).sort_values(["date", "code"]).reset_index(drop=True)


def _signal_start_date(
    month: str | None,
    month_features: pd.DataFrame | None,
    calendar: list[pd.Timestamp],
    entry_delay_days: int,
) -> pd.Timestamp | None:
    if month is None:
        return None
    if month_features is not None and "date" in month_features.columns and not month_features.empty:
        signal_date = pd.to_datetime(month_features["date"], errors="coerce").dropna().max()
    else:
        signal_date = pd.to_datetime(f"{month}-01", errors="coerce") + pd.offsets.MonthEnd(0)
    if pd.isna(signal_date):
        return None
    candidates = [date for date in calendar if date > pd.Timestamp(signal_date)]
    if not candidates:
        return None
    index = min(max(int(entry_delay_days), 1) - 1, len(candidates) - 1)
    return pd.Timestamp(candidates[index])


def _can_buy(row: pd.Series, exclude_st: bool, limit_threshold_pct: float) -> bool:
    if int(row.get("tradestatus", 1) or 0) != 1:
        return False
    if exclude_st and int(row.get("isST", 0) or 0) != 0:
        return False
    pct_chg = row.get("pctChg")
    if pd.notna(pct_chg) and float(pct_chg) >= float(limit_threshold_pct):
        return False
    return True


def _can_sell(row: pd.Series, limit_threshold_pct: float) -> bool:
    if int(row.get("tradestatus", 1) or 0) != 1:
        return False
    pct_chg = row.get("pctChg")
    if pd.notna(pct_chg) and float(pct_chg) <= -float(limit_threshold_pct):
        return False
    return True


def _entry_reason(
    daily: pd.DataFrame,
    day: pd.DataFrame,
    code: str,
    date: pd.Timestamp,
    day_index: int,
    observed_highs: dict[str, float],
    entry_trigger: str,
    entry_window_days: int,
    min_pullback_pct: float,
    max_pullback_pct: float,
    breakout_lookback_days: int,
    breakout_volume_multiplier: float,
) -> str | None:
    row = day.loc[code]
    close = float(row["close"])
    observed_highs[code] = max(float(observed_highs.get(code, close)), close)
    trigger = str(entry_trigger).strip().lower()
    if trigger in {"", "immediate"}:
        return "entry"
    if day_index >= int(entry_window_days):
        return None

    history = daily.loc[(daily["code"] == code) & (daily["date"] <= pd.Timestamp(date))].sort_values("date")
    previous_history = history.iloc[:-1]
    ma20 = history["close"].tail(20).mean()
    ma60 = history["close"].tail(60).mean()
    ret_20d = close / float(history["close"].iloc[-min(len(history), 20)]) - 1.0 if len(history) >= 2 else 0.0
    trend_ok = close >= float(ma20) and float(ma20) >= float(ma60) and ret_20d > 0.0
    if not trend_ok:
        return None

    drawdown = 1.0 - close / max(float(observed_highs.get(code, close)), close)
    prev_close = float(previous_history["close"].iloc[-1]) if not previous_history.empty else close
    pullback_ok = (
        float(min_pullback_pct) <= drawdown <= float(max_pullback_pct)
        and close >= float(ma20)
        and close > float(row.get("open", close))
        and close > prev_close
    )
    if trigger in {"pullback", "pullback_or_breakout"} and pullback_ok:
        return "pullback_entry"

    lookback = max(int(breakout_lookback_days), 1)
    breakout_base = previous_history.tail(lookback)
    if not breakout_base.empty:
        amount = float(row.get("amount", 0.0) or 0.0)
        amount_mean = float(breakout_base["amount"].mean()) if "amount" in breakout_base.columns else 0.0
        breakout_ok = (
            close > float(breakout_base["close"].max())
            and (amount_mean <= 0.0 or amount > amount_mean * float(breakout_volume_multiplier))
        )
        if trigger in {"breakout", "pullback_or_breakout"} and breakout_ok:
            return "breakout_entry"
    return None


def _trade_price(row: pd.Series, action: str, slippage_bps: float) -> float:
    price = float(row.get("open", row.get("close", 0.0)))
    slip = float(slippage_bps) / 10000.0
    return price * (1.0 + slip if action == "buy" else 1.0 - slip)


def _max_hold_hit(
    position: dict[str, float | pd.Timestamp],
    date: pd.Timestamp,
    max_hold_months: int,
) -> bool:
    if int(max_hold_months) <= 0:
        return False
    entry_date = pd.to_datetime(position.get("entry_date"), errors="coerce")
    if pd.isna(entry_date):
        return False
    exit_date = pd.Timestamp(entry_date) + pd.DateOffset(months=int(max_hold_months))
    return pd.Timestamp(date) >= exit_date


def _within_entry_batch(
    target_codes: list[str],
    code: str,
    day_index: int,
    entry_batch_days: int,
) -> bool:
    batch_days = max(int(entry_batch_days), 1)
    if batch_days <= 1:
        return True
    try:
        code_index = target_codes.index(code)
    except ValueError:
        return False
    batch_size = max((len(target_codes) + batch_days - 1) // batch_days, 1)
    batch_index = min(code_index // batch_size, batch_days - 1)
    return int(day_index) >= batch_index


def _append_trade_event(
    event_rows: list[dict[str, Any]],
    month: str,
    date: pd.Timestamp,
    code: str,
    action: str,
    reason: str,
    row: pd.Series,
    slippage_bps: float,
    weight: float = 0.0,
) -> None:
    event_rows.append(
        {
            "month": month,
            "date": pd.Timestamp(date),
            "code": code,
            "action": action,
            "reason": reason,
            "price": _trade_price(row, action, slippage_bps),
            "weight": float(weight),
        }
    )


def _format_event(date: pd.Timestamp, code: str, reason: str) -> str:
    return f"{pd.Timestamp(date).strftime('%Y-%m-%d')}:{code}:{reason}"


def _held_daily_return(holdings: dict[str, dict[str, float | pd.Timestamp]], day: pd.DataFrame) -> float:
    total_return = 0.0
    for code, position in holdings.items():
        if code not in day.index:
            continue
        row = day.loc[code]
        preclose = row.get("preclose")
        close = row.get("close")
        if pd.isna(preclose) or float(preclose) == 0.0 or pd.isna(close):
            continue
        weight = float(position.get("weight", 0.0))
        total_return += weight * (float(close) / float(preclose) - 1.0)
    return float(total_return)


def _target_buy_weight(target_codes: list[str], single_stock_max_weight: float) -> float:
    if not target_codes:
        return 0.0
    equal_weight = 1.0 / len(target_codes)
    cap = max(float(single_stock_max_weight), 0.0)
    return min(equal_weight, cap)


def _benchmark_returns(benchmark_df: pd.DataFrame, month: str) -> tuple[float, float]:
    if benchmark_df is None or benchmark_df.empty or "month" not in benchmark_df.columns:
        return 0.0, 0.0
    row = benchmark_df.loc[benchmark_df["month"].astype(str) == str(month)]
    if row.empty:
        return 0.0, 0.0
    hs300_column = "hs300_return_1m" if "hs300_return_1m" in row.columns else "benchmark_return_1m"
    zz500_column = "zz500_return_1m" if "zz500_return_1m" in row.columns else hs300_column
    return float(row.iloc[0].get(hs300_column, 0.0)), float(row.iloc[0].get(zz500_column, 0.0))


def _period_benchmark_returns(
    daily: pd.DataFrame,
    period_dates: list[pd.Timestamp],
    fallback_benchmark_df: pd.DataFrame,
    month: str,
) -> tuple[float, float]:
    hs300_return = _index_period_return(daily, "sh.000300", period_dates)
    zz500_return = _index_period_return(daily, "sh.000905", period_dates)
    fallback_hs300, fallback_zz500 = _benchmark_returns(fallback_benchmark_df, month)
    return (
        fallback_hs300 if hs300_return is None else hs300_return,
        fallback_zz500 if zz500_return is None else zz500_return,
    )


def _index_period_return(
    daily: pd.DataFrame,
    index_code: str,
    period_dates: list[pd.Timestamp],
) -> float | None:
    if daily.empty or not period_dates:
        return None
    index_daily = daily.loc[
        (daily["code"].astype(str) == index_code)
        & (daily["date"] >= pd.Timestamp(period_dates[0]))
        & (daily["date"] <= pd.Timestamp(period_dates[-1]))
    ].sort_values("date")
    if len(index_daily) < 2:
        return None
    first_close = pd.to_numeric(index_daily.iloc[0].get("close"), errors="coerce")
    last_close = pd.to_numeric(index_daily.iloc[-1].get("close"), errors="coerce")
    if pd.isna(first_close) or pd.isna(last_close) or float(first_close) == 0.0:
        return None
    return float(last_close) / float(first_close) - 1.0


def _compute_month_ic(month_df: pd.DataFrame) -> float:
    if len(month_df) < 2 or "future_stock_return_1m" not in month_df.columns:
        return 0.0
    correlation = month_df["score"].corr(month_df["future_stock_return_1m"])
    return 0.0 if pd.isna(correlation) else float(correlation)


def _empty_month_row(month: str, benchmark_df: pd.DataFrame, cumulative_net_value: float) -> dict[str, Any]:
    benchmark_hs300_return, benchmark_zz500_return = _benchmark_returns(benchmark_df, month)
    return {
        "month": month,
        "selected_count": 0,
        "portfolio_return": 0.0,
        "benchmark_return": benchmark_hs300_return,
        "benchmark_hs300_return": benchmark_hs300_return,
        "benchmark_zz500_return": benchmark_zz500_return,
        "excess_return": -benchmark_hs300_return,
        "excess_hs300_return": -benchmark_hs300_return,
        "excess_zz500_return": -benchmark_zz500_return,
        "turnover": 0.0,
        "transaction_cost": 0.0,
        "ic": 0.0,
        "cumulative_net_value": cumulative_net_value,
        "buy_codes": "",
        "sell_codes": "",
        "hold_codes": "",
        "buy_events": "",
        "sell_events": "",
    }
