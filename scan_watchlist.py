"""
Script quét tự động danh mục — chạy độc lập (không qua Streamlit), dùng cho
GitHub Actions lên lịch chạy mỗi sáng trước phiên giao dịch.

Kết quả lưu vào signals_latest.json — app.py có thể đọc file này để hiển thị
"kết quả quét sáng nay" mà không cần train lại khi bạn mở app.

Chạy thủ công: python scan_watchlist.py
"""

import json
import os
from datetime import datetime

import pandas as pd

from features import build_features
from models import quick_train_predict
from signals import add_signal_columns, compute_relative_strength
from sector_analysis import attach_score_changes, calculate_sector_rankings, sector_for_ticker
from market_context import analyze_market_context
from trade_plan import build_trade_plan
from paper_trading import process_journal
from news_utils import fetch_all_news
from smart_news import build_sector_news_scores, enrich_news

# Danh mục mặc định — VN30 (kỳ cơ cấu tháng 7/2026 của HOSE), sửa qua biến môi trường WATCHLIST nếu muốn khác
DEFAULT_WATCHLIST = [
    "ACB", "BID", "BSR", "BVH", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG",
    "MBB", "MSN", "MWG", "PLX", "POW", "SAB", "SSB", "SSI", "STB", "TCB",
    "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VPL", "VRE",
]

MIN_AVG_TRADE_VALUE = float(os.environ.get("MIN_AVG_TRADE_VALUE", 2_000_000_000))  # 2 tỷ đ/phiên mặc định


def fetch_history_standalone(ticker: str, days_back: int = 500):
    """Bản không phụ thuộc Streamlit (không dùng st.cache_data/st.secrets)."""
    from vnstock.api.quote import Quote
    from datetime import timedelta

    api_key = os.environ.get("VNSTOCK_API_KEY")
    if api_key:
        try:
            import vnai
            vnai.setup_api_key(api_key)
        except Exception:
            pass

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    for source in ["VCI", "MSN", "KBS"]:
        try:
            q = Quote(symbol=ticker, source=source)
            df = q.history(start=start, end=end, interval="1D")
            if df is not None and not df.empty:
                df = df.rename(columns={"time": "date"})
                df["date"] = pd.to_datetime(df["date"])
                return df.sort_values("date").reset_index(drop=True)
        except Exception:
            continue
    return None


def main():
    watchlist_env = os.environ.get("WATCHLIST")
    watchlist = [t.strip().upper() for t in watchlist_env.split(",")] if watchlist_env else DEFAULT_WATCHLIST

    print(f"Đang quét {len(watchlist)} mã: {watchlist}")
    vnindex_df = fetch_history_standalone("VNINDEX", 500)

    previous_sector_rankings = []
    try:
        with open("signals_latest.json", "r", encoding="utf-8") as old_file:
            previous_sector_rankings = json.load(old_file).get("sector_rankings", [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    try:
        with open("paper_trades.json", "r", encoding="utf-8") as journal_file:
            existing_journal = json.load(journal_file)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_journal = None

    results = []
    histories = {}
    for tk in watchlist:
        try:
            raw = fetch_history_standalone(tk, 500)
            if raw is None or raw.empty or len(raw) < 200:
                print(f"  {tk}: không đủ dữ liệu, bỏ qua")
                continue

            histories[tk] = raw
            feats = build_features(raw)
            feats_sig = add_signal_columns(feats)
            r1 = quick_train_predict(feats, "target_1")
            if r1 is None:
                continue

            avg_trade_value = float((raw["close"] * raw["volume"]).tail(20).mean())
            if avg_trade_value < MIN_AVG_TRADE_VALUE:
                print(f"  {tk}: bỏ qua — thanh khoản thấp (TB {avg_trade_value/1e9:.2f} tỷ đ/phiên)")
                continue

            rel_strength = None
            if vnindex_df is not None:
                rel_df = compute_relative_strength(raw, vnindex_df)
                if not rel_df.dropna().empty:
                    rel_strength = float(rel_df["rel_strength"].dropna().iloc[-1])

            results.append({
                "ticker": tk,
                "price": float(raw["close"].iloc[-1]),
                "change_pct": float((raw["close"].iloc[-1] / raw["close"].iloc[-2] - 1) * 100),
                "probability_t1": r1["probability"],
                "quick_accuracy": r1["quick_accuracy"],
                "volume_spike": bool(feats_sig["vol_spike"].iloc[-1]),
                "relative_strength": rel_strength,
                "avg_trade_value_bn": round(avg_trade_value / 1e9, 2),
            })
            print(f"  {tk}: xác suất T+1 = {r1['probability']*100:.0f}%")
        except Exception as e:
            print(f"  {tk}: lỗi — {e}")
            continue

    try:
        raw_news = fetch_all_news()
        enriched_news = enrich_news(raw_news)
        sector_news_scores = build_sector_news_scores(raw_news)
        news_digest = {
            "raw_count": int(len(raw_news)),
            "unique_count": int(len(enriched_news)),
            "sector_count": int(len(sector_news_scores)),
        }
    except Exception as exc:
        print(f"Không lấy được tin tức ngành: {exc}")
        sector_news_scores = {}
        news_digest = {"raw_count": 0, "unique_count": 0, "sector_count": 0}

    sector_rankings = attach_score_changes(
        calculate_sector_rankings(histories, vnindex_df, sector_news_scores),
        previous_sector_rankings,
    )

    sector_score_map = {row["sector"]: row["score"] for row in sector_rankings}
    market_context = analyze_market_context(vnindex_df)
    market_trend = market_context.get("trend") if market_context else None
    for row in results:
        ticker = row["ticker"]
        row["sector"] = sector_for_ticker(ticker)
        row["sector_score"] = sector_score_map.get(row["sector"])
        row["trade_plan"] = build_trade_plan(
            histories.get(ticker),
            sector_score=row["sector_score"],
            market_trend=market_trend,
        )

    scanned_at = datetime.now().isoformat()
    journal = process_journal(existing_journal, results, histories, scanned_at)

    output = {
        "schema_version": 5,
        "scanned_at": scanned_at,
        "watchlist": watchlist,
        "results": results,
        "sector_rankings": sector_rankings,
        "news_digest": news_digest,
    }

    with open("signals_latest.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open("paper_trades.json", "w", encoding="utf-8") as f:
        json.dump(journal, f, ensure_ascii=False, indent=2)

    print(
        f"Đã lưu {len(results)}/{len(watchlist)} mã, {len(sector_rankings)} nhóm ngành "
        f"và {journal['summary']['total_signals']} tín hiệu paper."
    )


if __name__ == "__main__":
    main()
