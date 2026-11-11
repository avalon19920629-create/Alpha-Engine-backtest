"""Phase 1: time-causal TTL90/Renew30 regression tests.

Research-only. Synthetic prices; never places trades or calls the network.
"""
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
import alpha_engine_backtest as engine


def _prices():
    dates = pd.bdate_range("2023-01-02", periods=250)
    n = np.arange(len(dates))
    return pd.DataFrame(
        {"AAA": 100 * np.exp(.003 * n), "BBB": 100 * np.exp(-.001 * n)},
        index=dates,
    )


def _variant(protocol="composite", ttl=90, extension=30):
    return {
        "name": "phase1_causal_fixture",
        "base_weight": .4,
        "residual_weight": .6,
        "total_holdings": 2,
        "us_holdings": 2,
        "jp_holdings": 0,
        "ttl_days": ttl,
        "renewal_protocol": protocol,
        "renewal_extension_days": extension,
    }


def _selected():
    return pd.DataFrame(
        {"Weight": [.7, .3], "Region": ["US", "US"]},
        index=["AAA", "BBB"],
    )


def _run(prices, health, variant=None):
    variant = variant or _variant()
    with patch.object(engine, "_select_ttl_candidates",
                      return_value=(_selected(), pd.DataFrame())), \
         patch.object(engine, "_health_row", side_effect=health):
        return engine.run_ttl_renewal_variant(
            prices, ["AAA", "BBB"], [], prices.index[1], prices.index[105],
            variant, detail_logs=False,
        )


def _signal(passed):
    return {
        "rank_pass": passed,
        "residual_pass": passed,
        "composite_pass": passed,
        "composite_count": 3 if passed else 0,
    }


def test_day90_failed_stock_keeps_all_precheck_returns():
    prices = _prices()
    day90 = engine._next_trade_date(
        prices.index, prices.index[1] + pd.Timedelta(days=90)
    )
    results = _run(
        prices, lambda _p, _us, _jp, _date, ticker, _region, _variant, _bm:
        _signal(ticker == "AAA")
    )
    returns, selection, _, _, _, _, decisions, _ = results
    source = prices.pct_change()
    for day in prices.loc[prices.index[1]:day90].index:
        expected = .7 * source.at[day, "AAA"] + .3 * source.at[day, "BBB"]
        assert np.isclose(returns.at[day], expected)
    following = prices.index[prices.index.get_loc(day90) + 1]
    assert np.isclose(returns.at[following], .7 * source.at[following, "AAA"])
    first_cycle = selection.loc[selection.trade_date == selection.trade_date.min()]
    assert pd.Timestamp(first_cycle.set_index("ticker").loc["BBB", "exit_date"]) == day90
    assert not decisions.set_index("ticker").loc["BBB", "renewed"]


def test_weekly_failure_only_changes_future_days():
    prices = _prices()
    first = prices.index[1]
    day90 = engine._next_trade_date(
        prices.index, first + pd.Timedelta(days=90)
    )
    final = engine._next_trade_date(
        prices.index, first + pd.Timedelta(days=120)
    )
    check = next(
        x for x in (
            engine._next_trade_date(prices.index, d)
            for d in pd.date_range(day90, final, freq="W-FRI")
        ) if x is not None and day90 < x <= final
    )
    returns, selection, _, _, _, _, _, events = _run(
        prices,
        lambda _p, _us, _jp, date, _ticker, _region, _variant, _bm:
        _signal(date == day90),
    )
    daily = prices.pct_change()
    for day in prices.loc[first:check].index:
        assert np.isclose(
            returns.at[day],
            .7 * daily.at[day, "AAA"] + .3 * daily.at[day, "BBB"],
        )
    after = prices.index[prices.index.get_loc(check) + 1]
    assert returns.at[after] == 0
    first_cycle = selection.loc[selection.trade_date == selection.trade_date.min()]
    assert all(pd.to_datetime(first_cycle.exit_date) == check)
    assert len(events.loc[events.event == "extension_degradation_end"]) == 2


def test_fixed_ttl_has_legacy_return_behavior():
    prices = _prices()
    returns, *_ = _run(
        prices,
        lambda *_args: pytest.fail("No health check in fixed TTL"),
        _variant(protocol=None),
    )
    until = engine._next_trade_date(
        prices.index, prices.index[1] + pd.Timedelta(days=90)
    )
    daily = prices.pct_change()
    for day in prices.loc[prices.index[1]:until].index:
        expected = .7 * daily.at[day, "AAA"] + .3 * daily.at[day, "BBB"]
        assert np.isclose(returns.at[day], expected)


def test_all_good_preserves_legacy_return_behavior():
    prices = _prices()
    returns, *_ = _run(prices, lambda *_args: _signal(True))
    until = engine._next_trade_date(
        prices.index, prices.index[1] + pd.Timedelta(days=120)
    )
    daily = prices.pct_change()
    for day in prices.loc[prices.index[1]:min(until, returns.index[-1])].index:
        expected = .7 * daily.at[day, "AAA"] + .3 * daily.at[day, "BBB"]
        assert np.isclose(returns.at[day], expected)


def test_renewal_extension_respects_configured_days():
    prices = _prices()
    result, selected, *_ = _run(
        prices, lambda *_args: _signal(True),
        _variant(protocol="rank", ttl=60, extension=15),
    )
    expected_end = engine._next_trade_date(
        prices.index, prices.index[1] + pd.Timedelta(days=75)
    )
    first_cycle = selected.loc[selected.trade_date == selected.trade_date.min()]
    assert all(pd.to_datetime(first_cycle.exit_date) == expected_end)
    assert result.at[expected_end] > 0


def test_future_health_changes_cannot_change_prior_earnings():
    prices = _prices()
    good, *_ = _run(prices, lambda *_args: _signal(True))
    bad, *_ = _run(prices, lambda *_args: _signal(False))
    until = engine._next_trade_date(
        prices.index, prices.index[1] + pd.Timedelta(days=90)
    )
    pd.testing.assert_series_equal(
        good.loc[prices.index[1]:until], bad.loc[prices.index[1]:until]
    )
