"""
App dự báo xác suất T+1 / T+3 cho cổ phiếu Việt Nam.
Chạy: streamlit run app.py

Kiến trúc phân tầng:
  vnstock/RSS (nguồn)  ->  fetch_history/fetch_all_news (kết nối)
  ->  features.py (chuẩn hoá)  ->  models.py (nghiệp vụ)  ->  app.py (giao diện)

LƯU Ý QUAN TRỌNG:
- Đây là công cụ THAM KHẢO, không phải khuyến nghị đầu tư.
- Xác suất/vùng giá phụ thuộc dữ liệu lịch sử, không phản ánh tin tức/sự kiện đột xuất.
- Luôn đối chiếu phần Kiểm định backtest trước khi tin vào con số.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from news_utils import fetch_all_news, get_news_for_ticker, summarize_sentiment
from features import build_features, FEATURE_COLS, FEATURE_LABELS
from models import (
    walk_forward_eval, confidence_label,
    train_quantile_models, predict_price_range,
    quick_train_predict,
)

st.set_page_config(page_title="Dự báo CK Việt Nam (tham khảo)", layout="wide")

# ---------------------------------------------------------------------------
# TẦNG KẾT NỐI — lấy dữ liệu thật
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def fetch_news_cached() -> pd.DataFrame:
    return fetch_all_news()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_history(ticker: str, days_back: int = 500) -> pd.DataFrame:
    """API mới của vnstock (vnstock.api.quote.Quote). Tự xác thực VNSTOCK_API_KEY
    nếu có trong secrets (tránh bị chặn 403). Thử lần lượt nhiều nguồn."""
    from vnstock.api.quote import Quote

    try:
        if "VNSTOCK_API_KEY" in st.secrets:
            import vnai
            vnai.setup_api_key(st.secrets["VNSTOCK_API_KEY"])
    except Exception:
        pass

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    last_errors = {}
    for source in ["VCI", "MSN", "KBS"]:
        try:
            q = Quote(symbol=ticker, source=source)
            df = q.history(start=start, end=end, interval="1D")
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "time": "date", "open": "open", "high": "high",
                    "low": "low", "close": "close", "volume": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except Exception as e:
            last_errors[source] = str(e)
            continue

    detail = " | ".join(f"{src}: {msg}" for src, msg in last_errors.items())
    raise ConnectionError(
        f"Không lấy được dữ liệu cho mã {ticker}. Nếu lỗi có '403'/'Forbidden', "
        f"cần đăng ký VNSTOCK_API_KEY tại vnstocks.com/login. Chi tiết: {detail}"
    )


# ---------------------------------------------------------------------------
# GIAO DIỆN
# ---------------------------------------------------------------------------

st.title("Dự báo xác suất T+1 / T+3 — Cổ phiếu Việt Nam")
st.caption("Công cụ tham khảo · dữ liệu thật qua vnstock · không phải khuyến nghị đầu tư")

st.warning(
    "⚠️ Đây là công cụ **tham khảo**, không đảm bảo chính xác. Mô hình chỉ học từ dữ liệu "
    "giá/khối lượng lịch sử — **không** biết trước tin tức, sự kiện doanh nghiệp, hay biến động "
    "vĩ mô đột xuất. Luôn xem phần Kiểm định backtest trước khi diễn giải xác suất/vùng giá.",
    icon="⚠️",
)

try:
    api_key_available = "ANTHROPIC_API_KEY" in st.secrets
except Exception:
    api_key_available = False

with st.sidebar:
    st.header("Cấu hình")
    ticker = st.text_input("Mã cổ phiếu (xem chi tiết)", value="VNM").strip().upper()
    days_back = st.slider("Số ngày dữ liệu lịch sử", 250, 1000, 500, step=50)
    run = st.button("Chạy dự báo chi tiết", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Xếp hạng nhiều mã")
    tickers_raw = st.text_area(
        "Danh sách mã (cách nhau bởi dấu phẩy)",
        value="VNM, VCB, HPG, FPT, VIC",
        height=80,
    )
    screen_run = st.button("Xếp hạng danh mục", use_container_width=True)

    st.divider()
    st.caption("Phân tích sentiment tin tức")
    use_claude_sentiment = st.toggle(
        "Dùng Claude API (chính xác hơn, tốn phí nhỏ)",
        value=False,
        disabled=not api_key_available,
        help="Cần cấu hình API key trong .streamlit/secrets.toml trước." if not api_key_available
             else "Ước tính ~$1-2/tháng với mức dùng thông thường.",
    )
    if not api_key_available:
        st.caption("⚠️ Chưa tìm thấy API key. Xem README để cấu hình.")


# ---------------------------------------------------------------------------
# BẢNG XẾP HẠNG NHIỀU MÃ
# ---------------------------------------------------------------------------

if screen_run:
    tickers_list = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers_list:
        st.warning("Nhập ít nhất 1 mã để xếp hạng.")
    elif len(tickers_list) > 15:
        st.warning("Giới hạn tối đa 15 mã mỗi lần để tránh chờ quá lâu.")
    else:
        st.subheader("📊 Bảng xếp hạng xác suất tăng (tham khảo nhanh)")
        st.caption(
            "⚠️ Bảng này dùng kiểm định **nhanh** (1 lần chia train/test), KHÔNG phải "
            "walk-forward đầy đủ như phần xem chi tiết 1 mã bên dưới. Dùng để rút gọn "
            "danh sách cần xem kỹ, không phải căn cứ quyết định cuối cùng."
        )

        progress = st.progress(0.0, text="Đang xử lý...")
        rows = []
        for i, tk in enumerate(tickers_list):
            progress.progress((i + 1) / len(tickers_list), text=f"Đang xử lý {tk}...")
            try:
                raw = fetch_history(tk, days_back=500)
                if raw.empty or len(raw) < 200:
                    continue
                feats = build_features(raw)
                r1 = quick_train_predict(feats, "target_1")
                r3 = quick_train_predict(feats, "target_3")
                if r1 is None or r3 is None:
                    continue
                last_price = raw["close"].iloc[-1]
                chg = (raw["close"].iloc[-1] / raw["close"].iloc[-2] - 1) * 100
                rows.append({
                    "Mã": tk, "Giá": f"{last_price:,.0f}", "% ngày": f"{chg:+.2f}%",
                    "Xác suất tăng T+1": r1["probability"],
                    "Xác suất tăng T+3": r3["probability"],
                    "Acc nhanh T+1": r1["quick_accuracy"],
                })
            except Exception as e:
                st.caption(f"Bỏ qua {tk}: {e}")
                continue
        progress.empty()

        if rows:
            rank_df = pd.DataFrame(rows).sort_values("Xác suất tăng T+1", ascending=False)
            display_df = rank_df.copy()
            display_df["Xác suất tăng T+1"] = display_df["Xác suất tăng T+1"].apply(lambda x: f"{x*100:.0f}%")
            display_df["Xác suất tăng T+3"] = display_df["Xác suất tăng T+3"].apply(lambda x: f"{x*100:.0f}%")
            display_df["Acc nhanh T+1"] = display_df["Acc nhanh T+1"].apply(lambda x: f"{x*100:.0f}%")
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            st.caption(
                "Mẹo: mã có 'Acc nhanh T+1' thấp gần 50% nghĩa là mô hình gần như đoán ngẫu nhiên "
                "cho mã đó — xác suất tăng của nó kém tin cậy hơn, dù con số có thể cao."
            )
        else:
            st.error("Không lấy được dữ liệu cho mã nào trong danh sách.")

    st.divider()


# ---------------------------------------------------------------------------
# XEM CHI TIẾT 1 MÃ
# ---------------------------------------------------------------------------

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
        current_price = float(last["close"])
        chg = (raw["close"].iloc[-1] / raw["close"].iloc[-2] - 1) * 100
        st.metric("Giá đóng cửa gần nhất", f"{current_price:,.0f} đ", f"{chg:+.2f}%")
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

    # -----------------------------------------------------------------
    # VÙNG GIÁ DỰ BÁO (quantile regression)
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("Vùng giá dự báo (tham khảo)")
    st.caption(
        "Khoảng giá phản ánh **đúng mức biến động thực tế** của mã trong quá khứ — "
        "có thể rộng, đây không phải lỗi mà là bản chất của dự báo ngắn hạn. "
        "Khoảng hẹp giả tạo sẽ gây hiểu lầm nguy hiểm hơn khoảng rộng trung thực."
    )

    with st.spinner("Đang tính vùng giá..."):
        qmodels_1 = train_quantile_models(feats, "fut_ret_1")
        qmodels_3 = train_quantile_models(feats, "fut_ret_3")

    if qmodels_1 and qmodels_3:
        range1 = predict_price_range(qmodels_1, latest_row, current_price)
        range3 = predict_price_range(qmodels_3, latest_row, current_price)

        rc1, rc2 = st.columns(2)
        for col, label, rg in [(rc1, "T+1", range1), (rc2, "T+3", range3)]:
            with col:
                st.markdown(f"**Vùng giá {label}**")
                st.markdown(
                    f"### {rg['lo_price']:,.0f} — {rg['hi_price']:,.0f} đ"
                )
                st.caption(
                    f"Trung vị: {rg['mid_price']:,.0f}đ ({rg['mid_pct']:+.1f}%) · "
                    f"Khoảng 10%-90% phân vị lịch sử"
                )
    else:
        st.info("Không đủ dữ liệu để tính vùng giá quantile cho mã này.")

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
        "xác suất/vùng giá ở trên nên được xem xét thận trọng hơn.",
        icon="ℹ️",
    )

    # -----------------------------------------------------------------
    # PANEL TIN TỨC
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

elif not screen_run:
    st.info("Nhập mã cổ phiếu ở thanh bên trái và bấm **Chạy dự báo chi tiết**, "
            "hoặc dùng **Xếp hạng danh mục** để lọc nhanh nhiều mã cùng lúc.")
