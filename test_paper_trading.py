import pandas as pd

from paper_trading import create_paper_trade, process_journal, summarize_journal, update_paper_trade


def _history(prices, lows=None, highs=None, start="2026-01-01"):
    lows = lows or [p * 0.99 for p in prices]
    highs = highs or [p * 1.01 for p in prices]
    return pd.DataFrame({
        "date": pd.date_range(start, periods=len(prices), freq="B"),
        "open": prices, "high": highs, "low": lows, "close": prices,
        "volume": [1_000_000] * len(prices),
    })


def _result(action="MUA THĂM DÒ"):
    return {
        "ticker": "FPT", "sector": "Công nghệ", "sector_score": 75,
        "probability_t1": 0.6, "quick_accuracy": 0.55,
        "trade_plan": {"action": action, "setup": "Pullback", "entry_reference": 100,
                       "stop_loss": 95, "tp1": 105, "tp2": 110},
    }


def test_only_actionable_signal_is_recorded():
    raw = _history([98, 99, 100])
    assert create_paper_trade(_result("CHỜ ĐIỂM MUA"), raw, "2026-01-05") is None
    assert create_paper_trade(_result(), raw, "2026-01-05")["id"].endswith(":FPT")


def test_future_horizons_do_not_use_signal_day():
    signal_raw = _history([98, 99, 100])
    trade = create_paper_trade(_result(), signal_raw, "2026-01-05")
    full = _history([98, 99, 100, 101, 102, 106, 107, 108])
    updated = update_paper_trade(trade, full)
    assert updated["horizon_returns_pct"]["t3"] == 5.7
    assert "t10" not in updated["horizon_returns_pct"]


def test_same_bar_stop_and_target_uses_conservative_stop():
    trade = create_paper_trade(_result(), _history([98, 99, 100]), "2026-01-05")
    full = _history([98, 99, 100, 100], lows=[97, 98, 99, 94], highs=[99, 100, 101, 111])
    updated = update_paper_trade(trade, full)
    assert updated["exit_reason"] == "stop_loss"
    assert updated["net_return_pct"] == -5.3


def test_journal_does_not_duplicate_same_ticker_and_date():
    raw = _history([98, 99, 100])
    first = process_journal(None, [_result()], {"FPT": raw}, "2026-01-05")
    second = process_journal(first, [_result()], {"FPT": raw}, "2026-01-05")
    assert len(second["trades"]) == 1


def test_summary_counts_only_available_evidence():
    summary = summarize_journal([])
    assert summary["total_signals"] == 0
    assert summary["win_rate_pct"] is None
