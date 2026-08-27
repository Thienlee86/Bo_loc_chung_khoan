"""
Tầng tín hiệu phát hiện sớm (signal detection) — kết hợp nhiều lớp tín hiệu
ĐỘC LẬP với nhau, và kiểm định bằng event-study thay vì chỉ tin vào 1 model.

Nguyên tắc: 1 tín hiệu đơn lẻ dễ nhiễu. Tín hiệu chỉ đáng chú ý khi NHIỀU lớp
độc lập cùng đồng thuận. Đây KHÔNG phải khuyến nghị mua/bán — chỉ là công cụ
tham khảo cá nhân, không thay thế phân tích của người có chứng chỉ hành nghề.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LỚP 1+2: TÍN HIỆU QUY TẮC (rule-based) — dùng được cho event-study vì tính
# nhanh trên toàn bộ lịch sử, không cần train lại model từng ngày
# ---------------------------------------------------------------------------

def add_signal_columns(feats: pd.DataFrame) -> pd.DataFrame:
    """Thêm cột tín hiệu tăng/giảm dạng quy tắc, dựa trên chỉ báo đã có sẵn
    trong features.py. Dùng cho cả hiển thị hiện tại lẫn event-study lịch sử."""
    d = feats.copy()

    rsi_prev = d["rsi14"].shift(1)
    d["rsi_cross_up"] = (rsi_prev < 50) & (d["rsi14"] >= 50)
    d["rsi_cross_down"] = (rsi_prev > 50) & (d["rsi14"] <= 50)

    d["vol_spike"] = d["vol_ratio"] > 1.5  # khối lượng bất thường: >1.5x trung bình 20 phiên

    # Tín hiệu tổng hợp quy tắc: RSI cắt lên 50 + khối lượng đột biến + xu hướng MA ủng hộ
    d["signal_bull"] = d["rsi_cross_up"] & d["vol_spike"] & (d["ma_cross"] > 0)
    d["signal_bear"] = d["rsi_cross_down"] & d["vol_spike"] & (d["ma_cross"] < 0)

    return d


def event_study(feats: pd.DataFrame, signal_col: str, ret_col: str) -> dict:
    """Kiểm định: mỗi lần tín hiệu xuất hiện trong quá khứ, giá thực tế biến
    động ra sao so với baseline (tất cả các phiên)? Đây là cách trung thực để
    biết tín hiệu có thật sự có 'edge' hay chỉ là trùng hợp."""
    data = feats.dropna(subset=[signal_col, ret_col])
    events = data[data[signal_col]]

    if len(events) < 5:
        return {"n_events": len(events), "insufficient": True}

    event_mean = float(events[ret_col].mean())
    event_pct_positive = float((events[ret_col] > 0).mean())

    baseline_mean = float(data[ret_col].mean())
    baseline_pct_positive = float((data[ret_col] > 0).mean())

    return {
        "n_events": len(events),
        "insufficient": False,
        "event_mean_return": event_mean,
        "event_pct_positive": event_pct_positive,
        "baseline_mean_return": baseline_mean,
        "baseline_pct_positive": baseline_pct_positive,
        "edge_return": event_mean - baseline_mean,
        "edge_pct_positive": event_pct_positive - baseline_pct_positive,
    }


# ---------------------------------------------------------------------------
# LỚP 3: SỨC MẠNH TƯƠNG ĐỐI so với VN-Index
# ---------------------------------------------------------------------------

def compute_relative_strength(stock_df: pd.DataFrame, index_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """So sánh % thay đổi giá của mã với VN-Index cùng kỳ. Trả về DataFrame
    có thêm cột 'rel_strength' (dương = mạnh hơn thị trường chung)."""
    s = stock_df[["date", "close"]].rename(columns={"close": "stock_close"})
    idx = index_df[["date", "close"]].rename(columns={"close": "index_close"})
    merged = pd.merge(s, idx, on="date", how="inner")

    stock_ret = merged["stock_close"].pct_change(window)
    index_ret = merged["index_close"].pct_change(window)
    merged["rel_strength"] = stock_ret - index_ret

    return merged[["date", "rel_strength"]]


# ---------------------------------------------------------------------------
# LỚP 4: TIN TỨC — quy đổi từ summarize_sentiment() đã có sẵn (news_utils.py)
# ---------------------------------------------------------------------------

def news_signal_direction(news_summary: dict) -> str:
    """news_summary là kết quả từ summarize_sentiment() trong news_utils.py."""
    if news_summary.get("n_news", 0) < 2:
        return "Chưa đủ tin để đánh giá"
    if news_summary["overall"] == "Nghiêng tích cực":
        return "Tích cực"
    if news_summary["overall"] == "Nghiêng tiêu cực":
        return "Tiêu cực"
    return "Trung lập"


# ---------------------------------------------------------------------------
# TỔNG HỢP 4 LỚP — chỉ gắn cờ "đáng chú ý" khi đa số đồng thuận
# ---------------------------------------------------------------------------

def composite_signal(ml_prob_1: float, latest_vol_spike: bool, latest_rel_strength, news_summary: dict) -> dict:
    """Tổng hợp 4 lớp tín hiệu độc lập, đếm phiếu đồng thuận tăng/giảm.
    latest_rel_strength có thể là None nếu không lấy được dữ liệu VN-Index."""
    votes_bull, votes_bear = 0, 0
    details = {}

    # Lớp 1: mô hình ML (đã có từ walk_forward_eval)
    if ml_prob_1 >= 0.55:
        votes_bull += 1
        details["Kỹ thuật (mô hình)"] = "🟢 Nghiêng tăng"
    elif ml_prob_1 <= 0.45:
        votes_bear += 1
        details["Kỹ thuật (mô hình)"] = "🔴 Nghiêng giảm"
    else:
        details["Kỹ thuật (mô hình)"] = "⚪ Trung tính"

    # Lớp 2: khối lượng bất thường (chỉ có ý nghĩa khi kết hợp chiều giá — ở đây
    # dùng đơn giản: có spike hay không, chiều dựa vào các lớp khác)
    if latest_vol_spike:
        details["Khối lượng"] = "🟡 Bất thường (khuếch đại tín hiệu khác)"
    else:
        details["Khối lượng"] = "⚪ Bình thường"

    # Lớp 3: sức mạnh tương đối vs VN-Index
    if latest_rel_strength is not None:
        if latest_rel_strength > 0.02:
            votes_bull += 1
            details["Sức mạnh tương đối"] = "🟢 Mạnh hơn VN-Index"
        elif latest_rel_strength < -0.02:
            votes_bear += 1
            details["Sức mạnh tương đối"] = "🔴 Yếu hơn VN-Index"
        else:
            details["Sức mạnh tương đối"] = "⚪ Tương đương thị trường"
    else:
        details["Sức mạnh tương đối"] = "❓ Không lấy được dữ liệu VN-Index"

    # Lớp 4: tin tức
    news_dir = news_signal_direction(news_summary)
    if news_dir == "Tích cực":
        votes_bull += 1
        details["Tin tức"] = "🟢 Tích cực"
    elif news_dir == "Tiêu cực":
        votes_bear += 1
        details["Tin tức"] = "🔴 Tiêu cực"
    else:
        details["Tin tức"] = f"⚪ {news_dir}"

    if votes_bull >= 3:
        verdict = "Nhiều tín hiệu đồng thuận TĂNG — đáng chú ý theo dõi thêm"
    elif votes_bear >= 3:
        verdict = "Nhiều tín hiệu đồng thuận GIẢM — đáng chú ý theo dõi thêm"
    else:
        verdict = "Tín hiệu chưa đủ đồng thuận — chưa có gì đặc biệt"

    return {
        "votes_bull": votes_bull, "votes_bear": votes_bear,
        "verdict": verdict, "details": details,
    }


# ---------------------------------------------------------------------------
# VÙNG CẮT LỖ / CHỐT LỜI THAM KHẢO (dựa trên ATR)
# ---------------------------------------------------------------------------

def compute_risk_levels(current_price: float, atr_pct: float, rr_ratio: float = 2.0, atr_multiplier: float = 1.5) -> dict:
    """Tính vùng cắt lỗ/chốt lời THAM KHẢO dựa trên biến động thực tế (ATR) của mã,
    không dựa trên cảm tính. atr_pct là ATR đã chuẩn hoá theo % giá (cột atr14 trong
    features.py). Đây là công cụ hỗ trợ quản trị rủi ro, KHÔNG phải khuyến nghị vào lệnh.

    Mặc định: khoảng cách cắt lỗ = 1.5x ATR, chốt lời = 2x khoảng cách cắt lỗ
    (tỷ lệ risk:reward = 1:2, một quy tắc phổ biến trong quản trị vốn)."""
    stop_distance_pct = atr_pct * atr_multiplier
    stop_loss = current_price * (1 - stop_distance_pct)
    take_profit = current_price * (1 + stop_distance_pct * rr_ratio)

    return {
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "stop_loss_pct": -stop_distance_pct * 100,
        "take_profit_pct": stop_distance_pct * rr_ratio * 100,
        "risk_reward_ratio": rr_ratio,
    }
    
