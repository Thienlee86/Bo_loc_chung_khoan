"""Bộ lọc nhanh toàn thị trường và phân nhóm cơ hội đầu tư."""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd


LIQUID_FALLBACK = [
    "ACB", "BID", "BSR", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
    "AAA", "ANV", "BAF", "BCM", "BMP", "CII", "CMG", "DBC", "DCM", "DGC",
    "DGW", "DIG", "DPM", "DXG", "EIB", "EVF", "FRT", "GEX", "GMD", "HAG",
    "HAH", "HCM", "HDG", "HHV", "HSG", "IDC", "KBC", "KDH", "KSB", "LPB",
    "MBS", "MSB", "NKG", "NLG", "NT2", "OCB", "ORS", "PAN", "PC1", "PDR",
    "PET", "PHR", "PVD", "PVS", "PVT", "REE", "SBT", "SHB", "SHS", "TCH",
    "VCG", "VCI", "VDS", "VGC", "VIX", "VND", "VOS", "VPI", "VSC", "VHC",
]


def _clean_symbols(values: Iterable[object]) -> list[str]:
    output = []
    for value in values:
        symbol = str(value or "").strip().upper()
        if re.fullmatch(r"[A-Z]{3}", symbol) and symbol not in output:
            output.append(symbol)
    return output


def symbols_from_listing(listing: pd.DataFrame | None) -> list[str]:
    if listing is None or listing.empty:
        return []
    symbol_col = next((c for c in ("symbol", "ticker", "code") if c in listing.columns), None)
    if symbol_col is None:
        return []
    frame = listing.copy()
    exchange_col = next((c for c in ("exchange", "exchange_name", "comGroupCode") if c in frame.columns), None)
    if exchange_col:
        allowed = {"HOSE", "HSX", "HNX", "UPCOM", "UPCOMINDEX"}
        exchange = frame[exchange_col].astype(str).str.upper().str.replace(" ", "", regex=False)
        valid = exchange.isin(allowed)
        if valid.any():
            frame = frame[valid]
    return _clean_symbols(frame[symbol_col].tolist())


def build_universe(listing: pd.DataFrame | None = None, override: str | None = None,
                   max_symbols: int = 120) -> list[str]:
    """Ưu tiên cấu hình tay; nếu không, ghép rổ thanh khoản với niêm yết động."""
    if override:
        return _clean_symbols(override.split(","))[:max_symbols]
    discovered = symbols_from_listing(listing)
    return _clean_symbols([*LIQUID_FALLBACK, *sorted(discovered)])[:max_symbols]


def fast_snapshot(ticker: str, history: pd.DataFrame,
                  min_avg_trade_value: float = 2_000_000_000) -> dict | None:
    required = {"close", "volume"}
    if history is None or len(history) < 60 or not required.issubset(history.columns):
        return None
    close = pd.to_numeric(history["close"], errors="coerce")
    volume = pd.to_numeric(history["volume"], errors="coerce")
    if close.tail(60).isna().any() or float(close.iloc[-1]) <= 0:
        return None
    avg_value = float((close * volume).tail(20).mean())
    if not np.isfinite(avg_value) or avg_value < min_avg_trade_value:
        return None
    ma20, ma50 = float(close.tail(20).mean()), float(close.tail(50).mean())
    volume20 = float(volume.tail(20).mean())
    return {
        "ticker": ticker,
        "price": float(close.iloc[-1]),
        "ret_5d_pct": float((close.iloc[-1] / close.iloc[-6] - 1) * 100),
        "ret_20d_pct": float((close.iloc[-1] / close.iloc[-21] - 1) * 100),
        "above_ma20": float(close.iloc[-1] > ma20),
        "above_ma50": float(close.iloc[-1] > ma50),
        "volume_ratio": float(volume.tail(5).mean() / volume20) if volume20 > 0 else 0.0,
        "avg_trade_value_bn": round(avg_value / 1e9, 2),
    }


def rank_fast_snapshots(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    frame = pd.DataFrame(rows).copy()
    for column in ("ret_5d_pct", "ret_20d_pct", "volume_ratio", "avg_trade_value_bn"):
        frame[f"_{column}"] = frame[column].rank(pct=True) * 100
    frame["fast_score"] = (
        frame["_ret_5d_pct"] * 0.15 + frame["_ret_20d_pct"] * 0.25
        + frame["above_ma20"] * 12.5 + frame["above_ma50"] * 12.5
        + frame["_volume_ratio"] * 0.20 + frame["_avg_trade_value_bn"] * 0.15
    ).clip(0, 100).round(1)
    columns = list(rows[0].keys()) + ["fast_score"]
    return frame.sort_values("fast_score", ascending=False)[columns].to_dict("records")


def opportunity_score(row: dict) -> float:
    plan = row.get("trade_plan") or {}
    quality = row.get("model_quality") or {}
    action_points = {
        "MUA THĂM DÒ": 90, "THEO DÕI BREAKOUT": 75, "CHỜ ĐIỂM MUA": 65,
        "CHỜ XÁC NHẬN": 55, "KHÔNG MUA ĐUỔI": 35, "KHÔNG MUA": 15,
    }
    score = (
        float(row.get("fast_score") or 0) * 0.40
        + float(row.get("probability_t1") or 0) * 100 * 0.25
        + float(row.get("sector_score") or 50) * 0.20
        + action_points.get(plan.get("action"), 45) * 0.15
    )
    score += {"PASS": 8, "CAUTION": 0, "BLOCK": -25}.get(quality.get("gate"), 0)
    return round(max(0, min(100, score)), 1)


def categorize_opportunities(results: list[dict], buy_limit: int = 5,
                             watch_limit: int = 10, avoid_limit: int = 5) -> dict:
    ranked = []
    for source in results:
        row = dict(source)
        row["opportunity_score"] = opportunity_score(row)
        ranked.append(row)
    ranked.sort(key=lambda item: item["opportunity_score"], reverse=True)
    buy = [r for r in ranked if (r.get("model_quality") or {}).get("gate") == "PASS"
           and (r.get("trade_plan") or {}).get("action") == "MUA THĂM DÒ"][:buy_limit]
    avoid = [r for r in reversed(ranked) if (r.get("model_quality") or {}).get("gate") == "BLOCK"
             or (r.get("trade_plan") or {}).get("action") == "KHÔNG MUA"][:avoid_limit]
    excluded = {r["ticker"] for r in buy + avoid}
    watch = [r for r in ranked if r.get("ticker") not in excluded][:watch_limit]
    return {"buy": buy, "watch": watch, "avoid": avoid}
