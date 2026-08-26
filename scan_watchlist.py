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

# Danh mục mặc định — sửa trực tiếp tại đây hoặc truyền qua biến môi trường WATCHLIST
DEFAULT_WATCHLIST = ["VNM", "VCB", "HPG", "FPT", "VIC", "VHM", "MSN", "MWG", "TCB", "GAS"]


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

    results = []
    for tk in watchlist:
        try:
            raw = fetch_history_standalone(tk, 500)
            if raw is None or raw.empty or len(raw) < 200:
                print(f"  {tk}: không đủ dữ liệu, bỏ qua")
                continue

            feats = build_features(raw)
            feats_sig = add_signal_columns(feats)
            r1 = quick_train_predict(feats, "target_1")
            if r1 is None:
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
            })
            print(f"  {tk}: xác suất T+1 = {r1['probability']*100:.0f}%")
        except Exception as e:
            print(f"  {tk}: lỗi — {e}")
            continue

    output = {
        "scanned_at": datetime.now().isoformat(),
        "watchlist": watchlist,
        "results": results,
    }

    with open("signals_latest.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Đã lưu kết quả vào signals_latest.json ({len(results)}/{len(watchlist)} mã thành công)")


if __name__ == "__main__":
    main()
