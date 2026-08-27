"""
Phân tích bối cảnh thị trường chung (VN-Index) — xu hướng + mức biến động hiện tại.

Mục đích: một tín hiệu "kỹ thuật tăng" ở 1 mã có ý nghĩa khác hẳn tuỳ theo
VN-Index đang tăng mạnh hay giảm mạnh. Module này tách riêng để hiển thị
SONG SONG với mọi tín hiệu khác, giúp người dùng tự đặt vào ngữ cảnh.
"""

import pandas as pd


def analyze_market_context(vnindex_df: pd.DataFrame) -> dict:
    """Phân tích xu hướng + biến động của VN-Index. Trả về None nếu không đủ dữ liệu."""
    if vnindex_df is None or vnindex_df.empty or len(vnindex_df) < 55:
        return None

    df = vnindex_df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ret"] = df["close"].pct_change()

    latest = df.iloc[-1]

    if latest["close"] > latest["ma20"] > latest["ma50"]:
        trend = "Xu hướng TĂNG"
        trend_icon = "🟢"
    elif latest["close"] < latest["ma20"] < latest["ma50"]:
        trend = "Xu hướng GIẢM"
        trend_icon = "🔴"
    else:
        trend = "Đi ngang / chưa rõ xu hướng"
        trend_icon = "⚪"

    vol20 = df["ret"].rolling(20).std()
    vol_series = vol20.dropna()
    if len(vol_series) >= 30:
        vol_percentile = float((vol_series.rank(pct=True)).iloc[-1])
        if vol_percentile > 0.8:
            vol_level = "Biến động CAO bất thường"
            vol_icon = "🔴"
        elif vol_percentile > 0.5:
            vol_level = "Biến động trên trung bình"
            vol_icon = "🟡"
        else:
            vol_level = "Biến động bình thường/thấp"
            vol_icon = "🟢"
    else:
        vol_percentile, vol_level, vol_icon = None, "Chưa đủ dữ liệu", "❓"

    change_5d = None
    if len(df) >= 6:
        change_5d = float((latest["close"] / df["close"].iloc[-6] - 1) * 100)

    return {
        "trend": trend, "trend_icon": trend_icon,
        "volatility_level": vol_level, "volatility_icon": vol_icon,
        "vol_percentile": vol_percentile,
        "latest_close": float(latest["close"]),
        "change_5d_pct": change_5d,
    }


def context_advisory_note(context: dict) -> str:
    """Gợi ý diễn giải ngắn — KHÔNG phải khuyến nghị, chỉ là lưu ý đọc tín hiệu đúng ngữ cảnh."""
    if context is None:
        return "Không lấy được dữ liệu VN-Index để đánh giá bối cảnh chung."

    notes = []
    if "GIẢM" in context["trend"]:
        notes.append("VN-Index đang trong xu hướng giảm — tín hiệu tăng ở từng mã nên xem xét thận trọng hơn vì dòng tiền chung yếu.")
    elif "TĂNG" in context["trend"]:
        notes.append("VN-Index đang trong xu hướng tăng — thuận lợi hơn cho các tín hiệu tăng ở từng mã.")

    if context["volatility_icon"] == "🔴":
        notes.append("Biến động thị trường cao bất thường — vùng giá/xác suất dự báo có độ tin cậy thấp hơn bình thường.")

    return " ".join(notes) if notes else "Bối cảnh thị trường chung không có gì đặc biệt."
