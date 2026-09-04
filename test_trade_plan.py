import numpy as np
import pandas as pd

from trade_plan import build_trade_plan, calculate_position_size


def _history(last_jump=0.0, last_volume_ratio=1.0, trend=0.15, n=80):
    close = np.linspace(100, 100 * (1 + trend), n)
    close[-1] *= 1 + last_jump
    volume = np.full(n, 1_000_000.0)
    volume[-1] *= last_volume_ratio
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="B"),
        "open": close * 0.995, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": volume,
    })


def test_trade_plan_has_ordered_risk_levels():
    plan = build_trade_plan(_history(), sector_score=70, market_trend="Xu hướng TĂNG")
    assert plan["stop_loss"] < plan["entry_reference"] < plan["tp1"] < plan["tp2"]
    assert plan["reward_risk_tp2"] == 2.0


def test_weak_sector_blocks_new_buy():
    plan = build_trade_plan(_history(), sector_score=35, market_trend="Đi ngang")
    assert plan["action"] == "KHÔNG MUA"


def test_extended_price_is_marked_as_chasing():
    plan = build_trade_plan(_history(last_jump=0.12, last_volume_ratio=2.0), sector_score=80)
    assert plan["is_chasing"] is True
    assert plan["action"] == "KHÔNG MUA ĐUỔI"


def test_position_size_respects_risk_and_weight_caps():
    position = calculate_position_size(100_000_000, 50_000, 47_500, 1.0, 15.0)
    assert position["quantity"] % 100 == 0
    assert position["position_value"] <= 15_000_000
    assert position["capital_at_risk"] <= 1_000_000


def test_invalid_position_has_zero_quantity():
    assert calculate_position_size(100_000_000, 50_000, 51_000)["quantity"] == 0
