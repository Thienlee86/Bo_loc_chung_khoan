"""
App dự báo xác suất T+1 / T+3 cho cổ phiếu Việt Nam.
Chạy: streamlit run app.py

LƯU Ý QUAN TRỌNG:
- Đây là công cụ THAM KHẢO, không phải khuyến nghị đầu tư.
- Xác suất mô hình đưa ra phụ thuộc vào dữ liệu lịch sử và không phản ánh
  các yếu tố tin tức/sự kiện đột xuất xảy ra sau thời điểm dự báo.
- Độ chính xác thực tế luôn cần đối chiếu với phần "Kiểm định backtest"
  trước khi tin vào bất kỳ con số nào.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from news_utils import fetch_all_news, get_news_for_ticker, summarize_sentiment

st.set_page_config(page_title="Dự báo CK Việt Nam (tham khảo)", layout="wide")

# ---------------------------------------------------------------------------
# 1. LẤY DỮ LIỆU THẬT (vnstock)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_cached() -> pd.DataFrame:
    """Cache 15 phút để không spam RSS mỗi lần rerun."""
    return fetch_all_news()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, days_back: int = 500) -> pd.DataFrame:
    from vnstock import Vnstock
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    stock = Vnstock().stock(symbol=ticker, source="VCI")
    df = stock.quote.history(start=start, end=end, interval="1D")
    df = df.rename(columns={
        "time": "date", "open": "open", "high": "high",
        "low": "low", "close": "close", "volume": "volume",
    })
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Đường trung bình & giao cắt
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma_cross"] = (d["ma5"] - d["ma20"]) / d["ma20"]

    # RSI(14)
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # Volume
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()

    # Biến động (ATR đơn giản hoá)
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean() / d["close"]

    # Bollinger %B
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    d["boll_pctb"] = (d["close"] - lower) / (upper - lower)

    # Lợi nhuận phiên trước (momentum ngắn)
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_3"] = d["close"].pct_change(3)

    # Nhãn: tăng/giảm sau 1 phiên và sau 3 phiên
    d["target_1"] = (d["close"].shift(-1) > d["close"]).astype(int)
    d["target_3"] = (d["close"].shift(-3) > d["close"]).astype(int)

    return d


FEATURE_COLS = ["ma_cross", "rsi14", "macd_hist", "vol_ratio", "atr14", "boll_pctb", "ret_1", "ret_3"]
FEATURE_LABELS = {
    "ma_cross": "Giao cắt MA5/MA20", "rsi14": "RSI(14)", "macd_hist": "MACD histogram",
    "vol_ratio": "Khối lượng / TB20", "atr14": "Biến động (ATR)", "boll_pctb": "Bollinger %B",
    "ret_1": "Lợi nhuận phiên trước", "ret_3": "Lợi nhuận 3 phiên trước",
}


# ---------------------------------------------------------------------------
# 3. MÔ HÌNH + WALK-FORWARD BACKTEST
# ---------------------------------------------------------------------------

def walk_forward_eval(d: pd.DataFrame, target_col: str, min_train: int = 150):
    """Đánh giá walk-forward: train trên quá khứ, dự báo từng điểm tương lai kế tiếp.
    Trả về accuracy mô hình, accuracy baseline (đoán theo xu hướng phiên trước), và model cuối cùng."""
    from lightgbm import LGBMClassifier

    data = d.dropna(subset=FEATURE_COLS + [target_col]).reset_index(drop=True)
    if len(data) < min_train + 20:
        return None

    preds, actuals, baseline_preds = [], [], []
    step = 5  # train lại mỗi 5 phiên để đỡ tốn thời gian, vẫn giữ tính walk-forward

    for i in range(min_train, len(data) - 1, step):
        train = data.iloc[:i]
        test = data.iloc[i:i + step]
        if len(test) == 0 or train[target_col].nunique() < 2:
            continue

        model = LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, verbose=-1,
        )
        model.fit(train[FEATURE_COLS], train[target_col])
        p = model.predict(test[FEATURE_COLS])

        preds.extend(p)
        actuals.extend(test[target_col].tolist())
        # baseline: đoán lặp lại xu hướng phiên gần nhất (ret_1 > 0 → đoán tăng)
        baseline_preds.extend((test["ret_1"] > 0).astype(int).tolist())

    if not preds:
        return None

    acc = float(np.mean(np.array(preds) == np.array(actuals)))
    baseline_acc = float(np.mean(np.array(baseline_preds) == np.array(actuals)))

    # Model cuối cùng huấn luyện trên toàn bộ dữ liệu để dự báo phiên tới
    final_model = LGBMClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
    )
    final_model.fit(data[FEATURE_COLS], data[target_col])

    return {
        "accuracy": acc, "baseline_accuracy": baseline_acc,
        "n_test": len(preds), "model": final_model,
    }


def confidence_label(acc: float, baseline: float) -> str:
    edge = acc - baseline
    if edge >= 0.06:
        return "cao"
    if edge >= 0.02:
        return "trung bình"
    return "thấp — gần bằng đoán ngẫu nhiên"


# ---------------------------------------------------------------------------
# 4. GIAO DIỆN
# ---------------------------------------------------------------------------

st.title("Dự báo xác suất T+1 / T+3 — Cổ phiếu Việt Nam")
st.caption("Công cụ tham khảo · dữ liệu thật qua vnstock · không phải khuyến nghị đầu tư")

st.warning(
    "⚠️ Đây là công cụ **tham khảo**, không đảm bảo chính xác. Mô hình chỉ học từ dữ liệu "
    "giá/khối lượng lịch sử — **không** biết trước tin tức, sự kiện doanh nghiệp, hay biến động "
    "vĩ mô đột xuất. Luôn xem phần Kiểm định backtest bên dưới trước khi diễn giải xác suất.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Cấu hình")
    ticker = st.text_input("Mã cổ phiếu", value="VNM").strip().upper()
    days_back = st.slider("Số ngày dữ liệu lịch sử", 250, 1000, 500, step=50)

    st.divider()
    st.caption("Phân tích sentiment tin tức")
    api_key_available = "ANTHROPIC_API_KEY" in st.secrets if hasattr(st, "secrets") else False
    use_claude_sentiment = st.toggle(
        "Dùng Claude API (chính xác hơn, tốn phí nhỏ)",
        value=False,
        disabled=not api_key_available,
        help="Cần cấu hình API key trong .streamlit/secrets.toml trước." if not api_key_available
             else "Ước tính ~$1-2/tháng với mức dùng thông thường.",
    )
    if not api_key_available:
        st.caption("⚠️ Chưa tìm thấy API key. Xem README để cấu hình.")

    run = st.button("Chạy dự báo", type="primary", use_container_width=True)

if run and ticker:
    with st.spinner(f"Đang lấy dữ liệu {ticker}..."):
        try:
            raw = fetch_history(ticker, days_back)
        except Exception as e:
            st.error(f"Không lấy được dữ liệu cho {ticker}. Lỗi: {e}")
            st.stop()

    if raw.empty or len(raw) < 200:
        st.error("Không đủ dữ liệu lịch sử để huấn luyện mô hình (cần tối thiểu ~200 phiên).")
        st.stop()

    feats = build_features(raw)

    col_price, col_info = st.columns([2, 1])
    with col_price:
        st.subheader(f"{ticker} — Diễn biến giá")
        st.line_chart(raw.set_index("date")["close"])
    with col_info:
        last = raw.iloc[-1]
        chg = (raw["close"].iloc[-1] / raw["close"].iloc[-2] - 1) * 100
        st.metric("Giá đóng cửa gần nhất", f"{last['close']:,.0f} đ", f"{chg:+.2f}%")
        st.metric("Khối lượng phiên gần nhất", f"{last['volume']:,.0f}")

    st.divider()
    st.subheader("Xác suất dự báo (tham khảo)")

    with st.spinner("Đang huấn luyện & kiểm định walk-forward (có thể mất 30-60 giây)..."):
        res1 = walk_forward_eval(feats, "target_1")
        res3 = walk_forward_eval(feats, "target_3")

    if res1 is None or res3 is None:
        st.error("Không đủ dữ liệu sạch để huấn luyện walk-forward. Thử tăng số ngày lịch sử.")
        st.stop()

    latest_row = feats.dropna(subset=FEATURE_COLS).iloc[[-1]]
    prob1 = res1["model"].predict_proba(latest_row[FEATURE_COLS])[0][1]
    prob3 = res3["model"].predict_proba(latest_row[FEATURE_COLS])[0][1]

    c1, c2 = st.columns(2)
    for col, label, prob, res in [(c1, "T+1 (phiên kế tiếp)", prob1, res1), (c2, "T+3 (3 phiên tới)", prob3, res3)]:
        with col:
            direction = "TĂNG" if prob >= 0.5 else "GIẢM"
            conf = confidence_label(res["accuracy"], res["baseline_accuracy"])
            st.markdown(f"**{label}**")
            st.progress(float(prob) if prob >= 0.5 else float(1 - prob))
            st.markdown(f"### {prob*100:.0f}% khả năng {direction}")
            st.caption(f"Độ tin cậy dự báo: **{conf}** · (dựa trên chênh lệch accuracy mô hình vs baseline)")

    st.divider()
    st.subheader("Kiểm định backtest (walk-forward)")
    st.caption("So sánh độ chính xác mô hình với baseline đơn giản (đoán lặp lại xu hướng phiên trước). "
               "Nếu mô hình không vượt baseline rõ rệt, độ tin cậy dự báo ở trên nên coi là thấp.")

    bt_df = pd.DataFrame({
        "Mốc": ["T+1", "T+3"],
        "Độ chính xác mô hình": [f"{res1['accuracy']*100:.1f}%", f"{res3['accuracy']*100:.1f}%"],
        "Baseline (xu hướng trước)": [f"{res1['baseline_accuracy']*100:.1f}%", f"{res3['baseline_accuracy']*100:.1f}%"],
        "Chênh lệch": [f"{(res1['accuracy']-res1['baseline_accuracy'])*100:+.1f}pp",
                       f"{(res3['accuracy']-res3['baseline_accuracy'])*100:+.1f}pp"],
        "Số phiên kiểm định": [res1["n_test"], res3["n_test"]],
    })
    st.dataframe(bt_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Yếu tố ảnh hưởng đến dự báo T+1")
    importances = pd.Series(res1["model"].feature_importances_, index=FEATURE_COLS)
    importances = importances / importances.sum()
    imp_df = pd.DataFrame({
        "Yếu tố": [FEATURE_LABELS[c] for c in importances.index],
        "Mức ảnh hưởng": importances.values,
    }).sort_values("Mức ảnh hưởng", ascending=True)
    st.bar_chart(imp_df.set_index("Yếu tố"))

    st.info(
        "Mô hình **không** đưa các yếu tố tin tức, công bố thông tin doanh nghiệp, hay biến động "
        "vĩ mô bất ngờ vào tính toán. Nếu có sự kiện lớn sắp diễn ra (họp ĐHCĐ, công bố KQKD, tin vĩ mô), "
        "xác suất ở trên nên được xem xét thận trọng hơn.",
        icon="ℹ️",
    )

    # -----------------------------------------------------------------
    # PANEL TIN TỨC — TÁCH BIỆT HOÀN TOÀN VỚI MÔ HÌNH/BACKTEST Ở TRÊN
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("📰 Tin tức liên quan (tham khảo — chưa đưa vào mô hình)")
    st.caption(
        "Sentiment tính bằng đếm từ khoá tích cực/tiêu cực (miễn phí, offline), "
        "**không hiểu ngữ cảnh/phủ định** — chỉ mang tính gợi ý sơ bộ, bạn nên tự đọc "
        "để đối chiếu với dự báo mô hình ở trên."
    )

    with st.spinner("Đang lấy tin tức mới nhất..."):
        try:
            news_df = fetch_news_cached()
        except Exception as e:
            news_df = pd.DataFrame()
            st.warning(f"Không lấy được tin tức lúc này. Lỗi: {e}")

    if not news_df.empty:
        claude_client = None
        if use_claude_sentiment and api_key_available:
            import anthropic
            claude_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

        ticker_news = get_news_for_ticker(ticker, news_df, use_claude=use_claude_sentiment, claude_client=claude_client)
        summary = summarize_sentiment(ticker_news)

        if summary["n_news"] == 0:
            st.write(f"Không tìm thấy tin nào nhắc trực tiếp đến **{ticker}** trong các nguồn RSS hiện tại.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tổng số tin", summary["n_news"])
            c2.metric("Tích cực", summary["n_pos"])
            c3.metric("Tiêu cực", summary["n_neg"])
            c4.metric("Xu hướng chung", summary["overall"])

            for _, row in ticker_news.iterrows():
                icon = {"Tích cực": "🟢", "Tiêu cực": "🔴", "Trung lập": "⚪"}[row["sentiment_label"]]
                with st.expander(f"{icon} {row['title']}"):
                    st.write(row["summary"] if row["summary"] else "_Không có tóm tắt._")
                    st.caption(f"Nguồn: {row['source']} · {row['published']}")
                    if row.get("reason"):
                        st.caption(f"Nhận định: {row['reason']}")
                    if row["matched_words"]:
                        st.caption(f"Từ khoá khớp: {', '.join(row['matched_words'])}")
                    st.link_button("Đọc bài gốc", row["link"])
    else:
        st.write("Chưa có dữ liệu tin tức để hiển thị.")

else:
    st.info("Nhập mã cổ phiếu ở thanh bên trái và bấm **Chạy dự báo** để bắt đầu.")
