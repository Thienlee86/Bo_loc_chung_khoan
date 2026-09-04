"""Lập kế hoạch giao dịch theo vùng giá và mức rủi ro, không dự báo chắc chắn."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _atr(raw: pd.DataFrame, window: int = 14) -> float:
    close = pd.to_numeric(raw["close"], errors="coerce")
    high = pd.to_numeric(raw["high"], errors="coerce")
    low = pd.to_numeric(raw["low"], errors="coerce")
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(window).mean().iloc[-1])


def _market_is_weak(market_trend: str | None) -> bool:
    return bool(market_trend and "GIẢM" in market_trend.upper())


def build_trade_plan(
    raw: pd.DataFrame,
    sector_score: float | None = None,
    market_trend: str | None = None,
) -> dict | None:
    """Tạo vùng mua, stop, TP1/TP2 và trạng thái hành động.

    Giá mua là một vùng, không phải một điểm. Stop ưu tiên hỗ trợ kỹ thuật gần
    nhất; nếu không có thì dùng 1.5 ATR. Kết quả chỉ là kế hoạch tham khảo.
    """
    if raw is None or raw.empty or len(raw) < 55:
        return None
    raw = raw.sort_values("date") if "date" in raw.columns else raw.copy()
    close = pd.to_numeric(raw["close"], errors="coerce")
    volume = pd.to_numeric(raw["volume"], errors="coerce")
    current = float(close.iloc[-1])
    atr = _atr(raw)
    if not np.isfinite(atr) or atr <= 0 or current <= 0:
        return None

    ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    prior_high20 = float(pd.to_numeric(raw["high"], errors="coerce").shift(1).rolling(20).max().iloc[-1])
    prior_low20 = float(pd.to_numeric(raw["low"], errors="coerce").shift(1).rolling(20).min().iloc[-1])
    avg_volume20 = float(volume.shift(1).rolling(20).mean().iloc[-1])
    volume_ratio = float(volume.iloc[-1] / avg_volume20) if avg_volume20 > 0 else 1.0

    breakout = current > prior_high20 and volume_ratio >= 1.2
    near_ema20 = abs(current - ema20) <= 0.6 * atr and current >= ema20 * 0.98
    if breakout:
        setup = "Breakout có khối lượng"
        entry_center = prior_high20
    elif near_ema20:
        setup = "Pullback quanh EMA20"
        entry_center = ema20
    else:
        setup = "Chờ giá về vùng hỗ trợ"
        candidates = [level for level in (ema20, prior_low20) if np.isfinite(level) and level < current]
        entry_center = max(candidates) if candidates else current - atr

    entry_low = max(0.0, entry_center - 0.30 * atr)
    entry_high = entry_center + 0.30 * atr
    chase_distance_atr = (current - entry_high) / atr
    is_chasing = chase_distance_atr > 0.5

    supports = [level for level in (ema20, prior_low20) if np.isfinite(level) and level < entry_low]
    if supports:
        stop_loss = max(supports) - 0.30 * atr
        stop_basis = "Hỗ trợ kỹ thuật − 0,3 ATR"
    else:
        stop_loss = entry_center - 1.50 * atr
        stop_basis = "Vùng mua − 1,5 ATR"
    stop_loss = max(0.0, min(stop_loss, entry_low - 0.10 * atr))

    risk_per_share = entry_center - stop_loss
    if risk_per_share <= 0:
        return None
    tp1 = entry_center + risk_per_share
    tp2 = entry_center + 2 * risk_per_share
    trailing_stop = max(ema20, float(close.tail(10).max()) - 2 * atr)

    if sector_score is not None and sector_score < 45:
        action = "KHÔNG MUA"
        reason = "Nhóm ngành đang yếu; ưu tiên bảo toàn vốn."
    elif _market_is_weak(market_trend):
        action = "CHỜ XÁC NHẬN"
        reason = "Thị trường chung đang giảm; chưa ưu tiên mở vị thế mới."
    elif is_chasing:
        action = "KHÔNG MUA ĐUỔI"
        reason = "Giá đã đi xa hơn vùng mua trên 0,5 ATR."
    elif entry_low <= current <= entry_high and (sector_score is None or sector_score >= 60):
        action = "MUA THĂM DÒ"
        reason = "Giá trong vùng mua và ngành không yếu."
    elif breakout and (sector_score is None or sector_score >= 60):
        action = "THEO DÕI BREAKOUT"
        reason = "Breakout có khối lượng; chờ retest hoặc tránh mua quá xa vùng kích hoạt."
    else:
        action = "CHỜ ĐIỂM MUA"
        reason = "Chưa đồng thời thỏa vị trí giá và sức mạnh ngành."

    return {
        "action": action, "reason": reason, "setup": setup,
        "current_price": round(current, 2), "entry_low": round(entry_low, 2),
        "entry_high": round(entry_high, 2), "entry_reference": round(entry_center, 2),
        "stop_loss": round(stop_loss, 2), "stop_basis": stop_basis,
        "tp1": round(tp1, 2), "tp2": round(tp2, 2),
        "trailing_stop": round(trailing_stop, 2), "atr": round(atr, 2),
        "atr_pct": round(atr / current * 100, 2),
        "risk_per_share": round(risk_per_share, 2),
        "risk_pct": round(risk_per_share / entry_center * 100, 2),
        "reward_risk_tp1": 1.0, "reward_risk_tp2": 2.0,
        "volume_ratio": round(volume_ratio, 2), "is_chasing": bool(is_chasing),
    }


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss: float,
    risk_percent: float = 1.0,
    max_position_percent: float = 15.0,
    lot_size: int = 100,
) -> dict:
    """Tính khối lượng theo rủi ro và giới hạn tỷ trọng, làm tròn xuống theo lô."""
    if capital <= 0 or entry_price <= stop_loss or risk_percent <= 0 or lot_size <= 0:
        return {"quantity": 0, "position_value": 0.0, "capital_at_risk": 0.0}
    risk_budget = capital * risk_percent / 100
    by_risk = math.floor((risk_budget / (entry_price - stop_loss)) / lot_size) * lot_size
    max_value = capital * max_position_percent / 100
    by_weight = math.floor((max_value / entry_price) / lot_size) * lot_size
    quantity = max(0, min(by_risk, by_weight))
    return {
        "quantity": int(quantity),
        "position_value": round(quantity * entry_price, 2),
        "capital_at_risk": round(quantity * (entry_price - stop_loss), 2),
        "risk_budget": round(risk_budget, 2),
        "limited_by": "Giới hạn tỷ trọng" if by_weight <= by_risk else "Ngân sách rủi ro",
    }
