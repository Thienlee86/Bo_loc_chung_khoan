"""Cập nhật snapshot giá trong phiên cho các ứng viên, không huấn luyện lại mô hình."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import pandas as pd


MAX_SYMBOLS = int(os.environ.get("INTRADAY_LIMIT", "30"))


def _first_value(row: pd.Series, names: tuple[str, ...]):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return None


def _normalize_quotes(frame: pd.DataFrame | None) -> dict[str, dict]:
    if frame is None or frame.empty:
        return {}
    symbol_col = next((c for c in ("symbol", "ticker", "code") if c in frame.columns), None)
    if symbol_col is None:
        return {}
    output = {}
    for _, row in frame.iterrows():
        ticker = str(row[symbol_col]).strip().upper()
        price = _first_value(row, ("match_price", "last_price", "price", "close"))
        reference = _first_value(row, ("reference_price", "ref_price", "prior_close"))
        change_pct = _first_value(row, ("change_percent", "change_pct", "percent_change"))
        if change_pct is None and price is not None and reference not in (None, 0):
            change_pct = (float(price) / float(reference) - 1) * 100
        if price is None:
            continue
        output[ticker] = {
            "price": float(price),
            "change_pct": float(change_pct) if change_pct is not None else None,
            "volume": float(_first_value(row, ("total_volume", "volume", "accumulated_volume")) or 0),
        }
    return output


def _fetch_batch(symbols: list[str]) -> dict[str, dict]:
    try:
        from vnstock import Market
        market = Market()
        return _normalize_quotes(market.quote(symbols))
    except Exception as exc:
        print(f"Không lấy được bảng giá theo lô: {exc}")
        return {}


def _fetch_one_by_one(symbols: list[str]) -> dict[str, dict]:
    output = {}
    try:
        from vnstock import Market
        market = Market()
    except Exception as exc:
        print(f"Không khởi tạo được Market: {exc}")
        return output
    for ticker in symbols:
        try:
            frame = market.equity(ticker).quote()
            normalized = _normalize_quotes(frame)
            if ticker in normalized:
                output[ticker] = normalized[ticker]
            elif len(normalized) == 1:
                output[ticker] = next(iter(normalized.values()))
        except Exception as exc:
            print(f"  {ticker}: lỗi snapshot — {exc}")
    return output


def main():
    with open("signals_latest.json", "r", encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)

    opportunities = data.get("opportunities") or {}
    rows = []
    for bucket in ("buy", "experimental", "watch"):
        rows.extend(opportunities.get(bucket) or [])
    symbols = []
    for row in rows:
        ticker = str(row.get("ticker") or "").upper()
        if ticker and ticker not in symbols:
            symbols.append(ticker)
    symbols = symbols[:MAX_SYMBOLS]

    now = datetime.now().isoformat()
    quotes = _fetch_batch(symbols)
    if not quotes and symbols:
        quotes = _fetch_one_by_one(symbols)

    data["intraday"] = {
        "updated_at": now,
        "session": "midday",
        "requested": len(symbols),
        "received": len(quotes),
        "quotes": quotes,
        "status": "ok" if quotes else "unavailable",
        "note": "Snapshot nguồn có thể trễ; kế hoạch kỹ thuật vẫn dựa trên dữ liệu ngày.",
    }
    with open("signals_latest.json", "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    print(f"Snapshot trong phiên: {len(quotes)}/{len(symbols)} mã.")


if __name__ == "__main__":
    main()
