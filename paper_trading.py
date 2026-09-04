"""Nhật ký paper trading bất biến và cập nhật kết quả bằng dữ liệu về sau."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


HORIZONS = (3, 5, 10, 20)
ELIGIBLE_ACTIONS = {"MUA THĂM DÒ"}


def _iso_date(value) -> str:
    return pd.Timestamp(value).date().isoformat()


def create_paper_trade(result: dict, raw: pd.DataFrame, created_at: str) -> dict | None:
    """Chỉ tạo lệnh khi app thực sự đánh dấu MUA THĂM DÒ."""
    plan = result.get("trade_plan")
    if not isinstance(plan, dict) or plan.get("action") not in ELIGIBLE_ACTIONS:
        return None
    if raw is None or raw.empty:
        return None
    signal_date = _iso_date(raw["date"].iloc[-1])
    ticker = result["ticker"]
    return {
        "id": f"{signal_date}:{ticker}", "ticker": ticker,
        "sector": result.get("sector"), "signal_date": signal_date,
        "created_at": created_at, "action": plan["action"], "setup": plan.get("setup"),
        "entry_price": float(plan["entry_reference"]),
        "stop_loss": float(plan["stop_loss"]), "tp1": float(plan["tp1"]), "tp2": float(plan["tp2"]),
        "sector_score": result.get("sector_score"),
        "probability_t1": result.get("probability_t1"),
        "quick_accuracy": result.get("quick_accuracy"),
        "status": "open", "tp1_hit": False, "tp2_hit": False, "stop_hit": False,
        "exit_date": None, "exit_price": None, "exit_reason": None,
        "net_return_pct": None, "mfe_pct": None, "mae_pct": None,
        "horizon_returns_pct": {},
    }


def update_paper_trade(trade: dict, raw: pd.DataFrame, round_trip_cost_pct: float = 0.30) -> dict:
    """Bổ sung kết quả chỉ từ các phiên sau ngày phát tín hiệu.

    Nếu stop và TP cùng xuất hiện trong một nến ngày, giả định stop xảy ra trước
    để tránh làm đẹp kết quả khi không có dữ liệu intraday.
    """
    item = deepcopy(trade)
    if raw is None or raw.empty or "date" not in raw:
        return item
    data = raw.copy()
    data["date"] = pd.to_datetime(data["date"])
    future = data[data["date"] > pd.Timestamp(item["signal_date"])].sort_values("date")
    if future.empty:
        return item

    entry = float(item["entry_price"])
    closes = pd.to_numeric(future["close"], errors="coerce")
    highs = pd.to_numeric(future["high"], errors="coerce")
    lows = pd.to_numeric(future["low"], errors="coerce")
    item["mfe_pct"] = round(float((highs.max() / entry - 1) * 100), 2)
    item["mae_pct"] = round(float((lows.min() / entry - 1) * 100), 2)

    for horizon in HORIZONS:
        key = f"t{horizon}"
        if len(future) >= horizon and key not in item["horizon_returns_pct"]:
            gross = (float(closes.iloc[horizon - 1]) / entry - 1) * 100
            item["horizon_returns_pct"][key] = round(gross - round_trip_cost_pct, 2)

    if item.get("status") == "open":
        for _, bar in future.iterrows():
            bar_date = _iso_date(bar["date"])
            if float(bar["low"]) <= float(item["stop_loss"]):
                item.update({
                    "status": "closed", "stop_hit": True, "exit_date": bar_date,
                    "exit_price": float(item["stop_loss"]), "exit_reason": "stop_loss",
                })
                break
            if float(bar["high"]) >= float(item["tp2"]):
                item.update({
                    "status": "closed", "tp1_hit": True, "tp2_hit": True,
                    "exit_date": bar_date, "exit_price": float(item["tp2"]), "exit_reason": "tp2",
                })
                break
            if float(bar["high"]) >= float(item["tp1"]):
                item["tp1_hit"] = True

        if item["status"] == "open" and len(future) >= 20:
            item.update({
                "status": "closed", "exit_date": _iso_date(future.iloc[19]["date"]),
                "exit_price": float(closes.iloc[19]), "exit_reason": "time_exit_t20",
            })

    if item.get("exit_price") is not None:
        gross = (float(item["exit_price"]) / entry - 1) * 100
        item["net_return_pct"] = round(gross - round_trip_cost_pct, 2)
    return item


def summarize_journal(trades: list[dict]) -> dict:
    closed = [trade for trade in trades if trade.get("status") == "closed" and trade.get("net_return_pct") is not None]
    returns = [float(trade["net_return_pct"]) for trade in closed]
    gross_profit = sum(value for value in returns if value > 0)
    gross_loss = abs(sum(value for value in returns if value < 0))
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else None

    horizon_metrics = {}
    for horizon in HORIZONS:
        values = [
            float(trade.get("horizon_returns_pct", {}).get(f"t{horizon}"))
            for trade in trades if trade.get("horizon_returns_pct", {}).get(f"t{horizon}") is not None
        ]
        horizon_metrics[f"t{horizon}"] = {
            "count": len(values),
            "win_rate_pct": round(sum(v > 0 for v in values) / len(values) * 100, 1) if values else None,
            "avg_return_pct": round(float(np.mean(values)), 2) if values else None,
        }

    return {
        "total_signals": len(trades), "open_signals": sum(t.get("status") == "open" for t in trades),
        "closed_signals": len(closed),
        "win_rate_pct": round(sum(v > 0 for v in returns) / len(returns) * 100, 1) if returns else None,
        "avg_net_return_pct": round(float(np.mean(returns)), 2) if returns else None,
        "profit_factor": profit_factor,
        "tp1_hit_rate_pct": round(sum(bool(t.get("tp1_hit")) for t in trades) / len(trades) * 100, 1) if trades else None,
        "stop_rate_pct": round(sum(bool(t.get("stop_hit")) for t in trades) / len(trades) * 100, 1) if trades else None,
        "horizons": horizon_metrics,
    }


def summarize_by_sector(trades: list[dict], horizon: int = 5) -> list[dict]:
    rows = []
    for trade in trades:
        value = trade.get("horizon_returns_pct", {}).get(f"t{horizon}")
        if value is not None:
            rows.append({"sector": trade.get("sector") or "Chưa phân loại", "return": float(value)})
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    output = frame.groupby("sector")["return"].agg(["count", "mean", lambda x: (x > 0).mean()]).reset_index()
    output.columns = ["sector", "count", "avg_return_pct", "win_rate"]
    output["avg_return_pct"] = output["avg_return_pct"].round(2)
    output["win_rate_pct"] = (output.pop("win_rate") * 100).round(1)
    return output.sort_values(["avg_return_pct", "count"], ascending=False).to_dict("records")


def process_journal(existing: dict | None, results: list[dict], histories: dict[str, pd.DataFrame], scanned_at: str) -> dict:
    existing = existing or {}
    trades = [update_paper_trade(t, histories.get(t.get("ticker"))) for t in existing.get("trades", [])]
    existing_ids = {trade["id"] for trade in trades}
    for result in results:
        trade = create_paper_trade(result, histories.get(result.get("ticker")), scanned_at)
        if trade and trade["id"] not in existing_ids:
            trades.append(trade)
            existing_ids.add(trade["id"])
    trades = sorted(trades, key=lambda item: (item["signal_date"], item["ticker"]), reverse=True)[:500]
    return {
        "schema_version": 1, "updated_at": scanned_at, "transaction_cost_pct": 0.30,
        "trades": trades, "summary": summarize_journal(trades),
        "sector_summary_t5": summarize_by_sector(trades, 5),
    }
