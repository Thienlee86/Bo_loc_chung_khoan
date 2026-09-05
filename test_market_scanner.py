import pandas as pd

from market_scanner import build_universe, categorize_opportunities, fast_snapshot, rank_fast_snapshots


def _history(multiplier=1.0):
    close = [100 + i * multiplier for i in range(80)]
    return pd.DataFrame({"close": close, "volume": [30_000_000] * 80})


def test_universe_prefers_override_and_cleans_symbols():
    assert build_universe(override=" hpg, fpt,HPG,INVALID ", max_symbols=10) == ["HPG", "FPT"]


def test_fast_rank_returns_best_first():
    strong = fast_snapshot("AAA", _history(1.0), 0)
    weak = fast_snapshot("BBB", _history(-0.2), 0)
    ranked = rank_fast_snapshots([weak, strong])
    assert ranked[0]["ticker"] == "AAA"
    assert ranked[0]["fast_score"] > ranked[1]["fast_score"]


def test_quality_gate_controls_buy_bucket():
    base = {"fast_score": 80, "probability_t1": 0.65, "sector_score": 70,
            "trade_plan": {"action": "MUA THĂM DÒ"}}
    rows = [
        {**base, "ticker": "AAA", "model_quality": {"gate": "PASS"}},
        {**base, "ticker": "BBB", "model_quality": {"gate": "CAUTION"}},
        {**base, "ticker": "CCC", "model_quality": {"gate": "BLOCK"}},
    ]
    groups = categorize_opportunities(rows)
    assert [r["ticker"] for r in groups["buy"]] == ["AAA"]
    assert "BBB" in [r["ticker"] for r in groups["watch"]]
    assert "CCC" in [r["ticker"] for r in groups["avoid"]]


def test_liquidity_converts_vnstock_thousand_dong_prices():
    history = pd.DataFrame({"close": [20.0] * 80, "volume": [1_000_000] * 80})
    snapshot = fast_snapshot("AAA", history, min_avg_trade_value=10_000_000_000)
    assert snapshot is not None
    assert snapshot["avg_trade_value_bn"] == 20.0
