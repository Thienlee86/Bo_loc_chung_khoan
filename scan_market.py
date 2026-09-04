"""Quét hai tầng: lọc nhanh toàn thị trường rồi phân tích sâu ứng viên tốt nhất."""

from __future__ import annotations

import json
import os
from datetime import datetime
import time

from features import build_features
from market_context import analyze_market_context
from market_scanner import build_universe, categorize_opportunities, fast_snapshot, rank_fast_snapshots
from model_monitor import attach_quality_reports, summarize_model_health
from models import quick_train_predict
from news_utils import fetch_all_news
from paper_trading import process_journal
from scan_watchlist import MIN_AVG_TRADE_VALUE, fetch_history_standalone
from sector_analysis import attach_score_changes, calculate_sector_rankings, sector_for_ticker
from signals import add_signal_columns, compute_relative_strength
from smart_news import build_sector_news_scores, enrich_news
from trade_plan import build_trade_plan


def _read_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _market_listing():
    """Tương thích nhiều bản Vnstock; lỗi nguồn sẽ tự dùng rổ dự phòng."""
    constructors = []
    try:
        from vnstock import Listing
        constructors.append(Listing)
    except Exception:
        pass
    try:
        from vnstock.api.listing import Listing
        constructors.append(Listing)
    except Exception:
        pass
    for constructor in constructors:
        for kwargs in ({"source": "VCI"}, {}):
            try:
                listing = constructor(**kwargs)
                data = listing.all_symbols()
                if data is not None and not data.empty:
                    return data
            except Exception:
                continue
    return None


def _fetch_universe(universe: list[str], request_interval: float) -> dict:
    """Điều tiết dưới giới hạn 20 request/phút của Vnstock Guest."""
    histories = {}
    for index, ticker in enumerate(universe):
        if index:
            time.sleep(request_interval)
        try:
            history = fetch_history_standalone(ticker, 500)
            if history is not None and not history.empty:
                histories[ticker] = history
                print(f"  {ticker}: {len(history)} phiên")
        except (Exception, SystemExit) as exc:
            print(f"  {ticker}: lỗi tải dữ liệu — {exc}")
    return histories


def main():
    universe_limit = int(os.environ.get("UNIVERSE_LIMIT", "120"))
    deep_limit = int(os.environ.get("DEEP_SCAN_LIMIT", "30"))
    request_interval = max(3.2, float(os.environ.get("REQUEST_INTERVAL_SECONDS", "3.2")))
    override = os.environ.get("WATCHLIST") or os.environ.get("MARKET_UNIVERSE")
    listing = None if override else _market_listing()
    universe = build_universe(listing, override, universe_limit)
    print(f"Tầng 1: quét nhanh {len(universe)} mã, giãn cách {request_interval:.1f} giây/yêu cầu")

    vnindex_df = fetch_history_standalone("VNINDEX", 500)
    histories = _fetch_universe(universe, request_interval)
    snapshots = []
    for ticker, history in histories.items():
        row = fast_snapshot(ticker, history, MIN_AVG_TRADE_VALUE)
        if row:
            snapshots.append(row)
    fast_ranking = rank_fast_snapshots(snapshots)
    finalists = [row["ticker"] for row in fast_ranking[:deep_limit]]
    fast_map = {row["ticker"]: row for row in fast_ranking}
    eligible_histories = {ticker: histories[ticker] for ticker in fast_map}
    print(f"Tầng 2: phân tích sâu {len(finalists)} mã: {finalists}")

    old_scan = _read_json("signals_latest.json", {})
    existing_journal = _read_json("paper_trades.json", None)
    try:
        raw_news = fetch_all_news()
        enriched_news = enrich_news(raw_news)
        sector_news_scores = build_sector_news_scores(raw_news)
        news_digest = {"raw_count": int(len(raw_news)), "unique_count": int(len(enriched_news)),
                       "sector_count": int(len(sector_news_scores))}
    except Exception as exc:
        print(f"Không lấy được tin tức: {exc}")
        sector_news_scores = {}
        news_digest = {"raw_count": 0, "unique_count": 0, "sector_count": 0}

    sector_rankings = attach_score_changes(
        calculate_sector_rankings(eligible_histories, vnindex_df, sector_news_scores),
        old_scan.get("sector_rankings", []),
    )
    sector_score_map = {row["sector"]: row["score"] for row in sector_rankings}
    market_context = analyze_market_context(vnindex_df)
    market_trend = market_context.get("trend") if market_context else None

    results = []
    for ticker in finalists:
        try:
            raw = histories[ticker]
            if len(raw) < 200:
                continue
            feats = build_features(raw)
            prediction = quick_train_predict(feats, "target_1")
            if prediction is None:
                continue
            signals = add_signal_columns(feats)
            rel_strength = None
            if vnindex_df is not None:
                rel = compute_relative_strength(raw, vnindex_df).dropna()
                if not rel.empty:
                    rel_strength = float(rel["rel_strength"].iloc[-1])
            sector = sector_for_ticker(ticker)
            item = {
                **fast_map[ticker], "ticker": ticker,
                "change_pct": float((raw["close"].iloc[-1] / raw["close"].iloc[-2] - 1) * 100),
                "probability_t1": prediction["probability"],
                "quick_accuracy": prediction["quick_accuracy"],
                "quick_baseline_accuracy": prediction["quick_baseline_accuracy"],
                "brier_score": prediction["brier_score"], "n_test": prediction["n_test"],
                "volume_spike": bool(signals["vol_spike"].iloc[-1]),
                "relative_strength": rel_strength, "sector": sector,
                "sector_score": sector_score_map.get(sector),
            }
            item["trade_plan"] = build_trade_plan(
                raw, sector_score=item["sector_score"], market_trend=market_trend)
            results.append(item)
            print(f"  {ticker}: nhanh {item['fast_score']:.1f}, xác suất T+1 {prediction['probability']*100:.0f}%")
        except Exception as exc:
            print(f"  {ticker}: lỗi phân tích — {exc}")

    scanned_at = datetime.now().isoformat()
    journal = process_journal(existing_journal, results, histories, scanned_at)
    results = attach_quality_reports(results, journal.get("trades", []))
    opportunities = categorize_opportunities(results)
    output = {
        "schema_version": 8, "scanned_at": scanned_at, "watchlist": universe,
        "universe_stats": {"requested": len(universe), "downloaded": len(histories),
                           "liquid": len(fast_ranking), "deep_analyzed": len(results)},
        "fast_ranking": fast_ranking[:50], "opportunities": opportunities,
        "results": results, "sector_rankings": sector_rankings, "news_digest": news_digest,
        "model_health": summarize_model_health(results),
    }
    with open("signals_latest.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    with open("paper_trades.json", "w", encoding="utf-8") as handle:
        json.dump(journal, handle, ensure_ascii=False, indent=2)
    print(f"Hoàn tất: {len(fast_ranking)} mã đạt thanh khoản, {len(results)} mã phân tích sâu, "
          f"{len(opportunities['buy'])} mua / {len(opportunities['watch'])} theo dõi / "
          f"{len(opportunities['avoid'])} tránh.")


if __name__ == "__main__":
    main()
