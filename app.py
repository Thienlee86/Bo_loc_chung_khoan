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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from news_utils import fetch_all_news, get_news_for_ticker, summarize_sentiment
from features import build_features, FEATURE_COLS, FEATURE_LABELS
from models import (
    walk_forward_eval, confidence_label,
    train_quantile_models, predict_price_range,
    quick_train_predict,
)
from signals import (
    add_signal_columns, event_study, compute_relative_strength,
    composite_signal, compute_risk_levels,
)
from market_context import analyze_market_context, context_advisory_note
from sector_analysis import sector_for_ticker
from trade_plan import build_trade_plan, calculate_position_size

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
    nếu có trong secrets (tránh bị chặn 403). Thử lần lượt nhiều nguồn, mỗi nguồn
    có TIMEOUT CỨNG 15 giây — nếu 1 nguồn bị treo (không lỗi, không trả kết quả),
    tự chuyển sang nguồn tiếp theo thay vì làm treo cả app."""
    from vnstock.api.quote import Quote

    try:
        if "VNSTOCK_API_KEY" in st.secrets:
            import vnai
            vnai.setup_api_key(st.secrets["VNSTOCK_API_KEY"])
    except Exception:
        pass

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    def _call_source(source: str):
        q = Quote(symbol=ticker, source=source)
        return q.history(start=start, end=end, interval="1D")

    last_errors = {}
    for source in ["VCI", "MSN", "KBS"]:
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call_source, source)
                df = future.result(timeout=15)
            if df is not None and not df.empty:
                df = df.rename(columns={
                    "time": "date", "open": "open", "high": "high",
                    "low": "low", "close": "close", "volume": "volume",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except FutureTimeoutError:
            last_errors[source] = "Timeout sau 15 giây — nguồn không phản hồi"
            continue
        except Exception as e:
            last_errors[source] = str(e)
            continue

    detail = " | ".join(f"{src}: {msg}" for src, msg in last_errors.items())
    raise ConnectionError(
        f"Không lấy được dữ liệu cho mã {ticker}. Nếu lỗi có '403'/'Forbidden', "
        f"cần đăng ký VNSTOCK_API_KEY tại vnstocks.com/login. Chi tiết: {detail}"
    )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_vnindex(days_back: int = 500):
    """Lấy dữ liệu VN-Index để tính sức mạnh tương đối. Trả về None nếu lỗi —
    không nên làm sập cả app chỉ vì thiếu 1 lớp tín hiệu phụ."""
    try:
        return fetch_history("VNINDEX", days_back)
    except Exception:
        return None


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
    if st.button("Điền nhanh 15 mã đầu VN30", use_container_width=True):
        st.session_state["tickers_raw_value"] = "ACB, BID, BSR, BVH, CTG, FPT, GAS, GVR, HDB, HPG, MBB, MSN, MWG, PLX, POW"
    tickers_raw = st.text_area(
        "Danh sách mã (cách nhau bởi dấu phẩy)",
        value=st.session_state.get("tickers_raw_value", "VNM, VCB, HPG, FPT, VIC"),
        height=80,
        key="tickers_raw_value",
    )
    min_trade_value = st.slider(
        "Lọc thanh khoản tối thiểu (tỷ đ/phiên)", 0.0, 10.0, 2.0, step=0.5,
        help="Mã có giá trị giao dịch trung bình 20 phiên thấp hơn mức này sẽ bị loại khỏi bảng xếp hạng.",
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
# KẾT QUẢ QUÉT TỰ ĐỘNG — MÀN HÌNH CHÍNH (Giai đoạn 1: đẩy lên hàng đầu)
# ---------------------------------------------------------------------------

import json as _json
import os as _os

st.markdown("## 🌅 Bảng tin sáng nay — VN30")

try:
    with st.spinner("Đang lấy bối cảnh thị trường chung (VN-Index)..."):
        vnindex_top = fetch_vnindex(300)
    market_ctx = analyze_market_context(vnindex_top)
except Exception:
    market_ctx = None

if market_ctx:
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Xu hướng VN-Index", f"{market_ctx['trend_icon']} {market_ctx['trend']}")
    mc2.metric("Mức biến động", f"{market_ctx['volatility_icon']} {market_ctx['volatility_level']}")
    mc3.metric("VN-Index thay đổi 5 phiên", f"{market_ctx['change_5d_pct']:+.2f}%" if market_ctx['change_5d_pct'] is not None else "N/A")
    st.caption(f"💡 {context_advisory_note(market_ctx)}")
else:
    st.caption("Không lấy được bối cảnh VN-Index — các tín hiệu bên dưới nên được đọc độc lập, thận trọng hơn.")

st.divider()

if _os.path.exists("signals_latest.json"):
    try:
        with open("signals_latest.json", "r", encoding="utf-8") as f:
            scan_data = _json.load(f)
        st.caption(
            f"Quét lúc: {scan_data['scanned_at']} · {len(scan_data['results'])} mã đạt tiêu chuẩn "
            f"thanh khoản (trên tổng {len(scan_data.get('watchlist', []))} mã trong danh mục VN30)"
        )

        sector_rows = scan_data.get("sector_rankings", [])
        if sector_rows:
            st.markdown("### 🧭 Sức mạnh nhóm ngành")
            status_icons = {
                "Dẫn dắt mạnh": "🟢", "Đang cải thiện": "🔵",
                "Trung tính / tích lũy": "⚪", "Suy yếu": "🟡",
                "Điều chỉnh mạnh": "🔴",
            }
            sector_df = pd.DataFrame(sector_rows)
            sector_df["status"] = sector_df["status"].apply(
                lambda value: f"{status_icons.get(value, '⚪')} {value}"
            )
            sector_df["score_change"] = sector_df["score_change"].apply(
                lambda value: "Mới" if pd.isna(value) else f"{value:+.1f}"
            )
            sector_df["return_5d_pct"] = sector_df["return_5d_pct"].apply(lambda value: f"{value:+.1f}%")
            sector_df["relative_strength_20d_pct"] = sector_df["relative_strength_20d_pct"].apply(
                lambda value: f"{value:+.1f}%"
            )
            sector_df["breadth_ma20_pct"] = sector_df["breadth_ma20_pct"].apply(lambda value: f"{value:.0f}%")
            sector_df["volume_ratio"] = sector_df["volume_ratio"].apply(lambda value: f"{value:.2f}x")
            sector_df["news_score"] = sector_df["news_score"].apply(
                lambda value: "Chưa có" if pd.isna(value) else f"{value:.0f}/100"
            )
            sector_df["news_count"] = sector_df["news_count"].apply(lambda value: f"{int(value)} tin")
            sector_df = sector_df[[
                "sector", "score", "score_change", "status", "return_5d_pct",
                "relative_strength_20d_pct", "breadth_ma20_pct", "volume_ratio",
                "news_score", "news_count",
            ]].rename(columns={
                "sector": "Nhóm ngành", "score": "Điểm", "score_change": "Thay đổi",
                "status": "Trạng thái", "return_5d_pct": "Lợi nhuận 5P",
                "relative_strength_20d_pct": "Mạnh/yếu 20P",
                "breadth_ma20_pct": "% mã trên MA20", "volume_ratio": "KL/TB20",
                "news_score": "Điểm tin", "news_count": "Bằng chứng",
            })
            st.dataframe(sector_df, hide_index=True, use_container_width=True)
            with st.expander("Cách đọc điểm ngành"):
                st.write(
                    "Khi ngành có tin: kỹ thuật chiếm 90% và tin tức chiếm 10%; khi chưa có tin, "
                    "app giữ nguyên điểm kỹ thuật, không tự gán tin trung lập. Điểm tin đã loại bài trùng, "
                    "giảm trọng số tin cũ và tăng trọng số cho sự kiện quan trọng. "
                    "Cột Thay đổi so với lần quét trước giúp phát hiện ngành đang tăng tốc hoặc suy yếu. "
                    "Ngành chỉ có một mã đại diện cần được đọc thận trọng hơn."
                )
        else:
            st.info("Chưa có dữ liệu ngành. Kết quả sẽ xuất hiện sau lần quét tự động tiếp theo.")

        st.markdown("### 📋 Cổ phiếu trong danh mục")
        if scan_data["results"]:
            scan_df = pd.DataFrame(scan_data["results"]).sort_values("probability_t1", ascending=False)
            scan_df["probability_t1"] = scan_df["probability_t1"].apply(lambda x: f"{x*100:.0f}%")
            scan_df["quick_accuracy"] = scan_df["quick_accuracy"].apply(lambda x: f"{x*100:.0f}%")
            scan_df["volume_spike"] = scan_df["volume_spike"].apply(lambda x: "🟡 Có" if x else "⚪ Không")
            scan_df["relative_strength"] = scan_df["relative_strength"].apply(
                lambda x: f"{x*100:+.1f}%" if x is not None else "N/A"
            )
            if "avg_trade_value_bn" in scan_df.columns:
                scan_df["avg_trade_value_bn"] = scan_df["avg_trade_value_bn"].apply(lambda x: f"{x:,.1f} tỷ")
            scan_df["trade_action"] = scan_df["trade_plan"].apply(
                lambda plan: plan.get("action", "Chưa có") if isinstance(plan, dict) else "Chưa có"
            )
            scan_df["entry_zone"] = scan_df["trade_plan"].apply(
                lambda plan: f"{plan['entry_low']:,.0f}–{plan['entry_high']:,.0f}" if isinstance(plan, dict) else "N/A"
            )
            scan_df["stop_level"] = scan_df["trade_plan"].apply(
                lambda plan: f"{plan['stop_loss']:,.0f}" if isinstance(plan, dict) else "N/A"
            )
            scan_df["tp2_level"] = scan_df["trade_plan"].apply(
                lambda plan: f"{plan['tp2']:,.0f}" if isinstance(plan, dict) else "N/A"
            )
            scan_df = scan_df.drop(columns=["trade_plan"], errors="ignore")
            scan_df = scan_df.rename(columns={
                "ticker": "Mã", "price": "Giá", "change_pct": "% ngày",
                "probability_t1": "Xác suất T+1", "quick_accuracy": "Acc nhanh",
                "volume_spike": "KL bất thường", "relative_strength": "Mạnh/yếu vs VN-Index",
                "avg_trade_value_bn": "GTGD TB/phiên", "sector": "Ngành",
                "sector_score": "Điểm ngành", "trade_action": "Trạng thái",
                "entry_zone": "Vùng mua", "stop_level": "Cắt lỗ", "tp2_level": "TP2",
            })
            st.dataframe(scan_df, hide_index=True, use_container_width=True)
            st.caption(
                "Đã tự động loại các mã có giá trị giao dịch trung bình dưới 2 tỷ đ/phiên "
                "(thanh khoản quá thấp, tín hiệu kỹ thuật dễ bị nhiễu). "
                "Đây là kết quả quét nhanh — bấm 'Chạy dự báo chi tiết' ở mã bạn quan tâm để xem đầy đủ "
                "backtest, vùng giá, và kiểm định tín hiệu trước khi cân nhắc."
            )
        else:
            st.info("Lần quét gần nhất không có mã nào đạt tiêu chuẩn thanh khoản.")
    except Exception:
        st.warning("File kết quả quét bị lỗi hoặc chưa đầy đủ. Thử chạy lại workflow trên GitHub Actions.")
else:
    st.info(
        "**Chưa có kết quả quét tự động.** App đang ở chế độ chờ — bạn cần bật tính năng quét hằng ngày "
        "qua GitHub Actions (xem mục 'Tự động quét mỗi sáng' trong README) để bảng này tự có dữ liệu mỗi "
        "khi bạn mở app, không cần tự bấm nút. Trong lúc chờ, bạn vẫn dùng được phần 'Xếp hạng danh mục' "
        "hoặc 'Chạy dự báo chi tiết' ở sidebar bên trái."
    )

st.divider()
st.markdown("## 🧪 Kiểm định paper trading")

if _os.path.exists("paper_trades.json"):
    try:
        with open("paper_trades.json", "r", encoding="utf-8") as paper_file:
            paper_data = _json.load(paper_file)
        paper_summary = paper_data.get("summary", {})
        pm1, pm2, pm3, pm4 = st.columns(4)
        pm1.metric("Tổng tín hiệu", paper_summary.get("total_signals", 0))
        pm2.metric("Đang theo dõi", paper_summary.get("open_signals", 0))
        win_rate = paper_summary.get("win_rate_pct")
        avg_return = paper_summary.get("avg_net_return_pct")
        pm3.metric("Tỷ lệ thắng đã đóng", "Chưa đủ mẫu" if win_rate is None else f"{win_rate:.1f}%")
        pm4.metric("Lợi nhuận ròng TB", "Chưa đủ mẫu" if avg_return is None else f"{avg_return:+.2f}%")

        horizons = paper_summary.get("horizons", {})
        horizon_rows = []
        for key, label in [("t3", "T+3"), ("t5", "T+5"), ("t10", "T+10"), ("t20", "T+20")]:
            metric = horizons.get(key, {})
            horizon_rows.append({
                "Mốc": label,
                "Số mẫu": metric.get("count", 0),
                "Tỷ lệ tăng": "Chưa đủ" if metric.get("win_rate_pct") is None else f"{metric['win_rate_pct']:.1f}%",
                "Lợi nhuận ròng TB": "Chưa đủ" if metric.get("avg_return_pct") is None else f"{metric['avg_return_pct']:+.2f}%",
            })
        st.dataframe(pd.DataFrame(horizon_rows), hide_index=True, use_container_width=True)
        st.caption(
            f"Đã giả định tổng chi phí mua–bán {paper_data.get('transaction_cost_pct', 0.30):.2f}%. "
            "Chỉ tín hiệu MUA THĂM DÒ mới được ghi. Nếu stop và TP cùng chạm trong một nến ngày, "
            "app tính stop trước để tránh thiên lệch có lợi."
        )

        paper_trades = paper_data.get("trades", [])
        if paper_trades:
            recent_rows = []
            for trade in paper_trades[:20]:
                returns = trade.get("horizon_returns_pct", {})
                recent_rows.append({
                    "Ngày": trade.get("signal_date"), "Mã": trade.get("ticker"),
                    "Ngành": trade.get("sector"), "Trạng thái": trade.get("status"),
                    "Giá vào": trade.get("entry_price"), "Stop": trade.get("stop_loss"),
                    "T+5": "Chờ" if returns.get("t5") is None else f"{returns['t5']:+.2f}%",
                    "T+20": "Chờ" if returns.get("t20") is None else f"{returns['t20']:+.2f}%",
                    "Kết quả lệnh": "Đang mở" if trade.get("net_return_pct") is None else f"{trade['net_return_pct']:+.2f}%",
                    "Lý do thoát": trade.get("exit_reason") or "—",
                })
            with st.expander("Xem 20 tín hiệu paper gần nhất"):
                st.dataframe(pd.DataFrame(recent_rows), hide_index=True, use_container_width=True)

        sector_paper = paper_data.get("sector_summary_t5", [])
        if sector_paper:
            with st.expander("Hiệu quả T+5 theo nhóm ngành"):
                sector_paper_df = pd.DataFrame(sector_paper).rename(columns={
                    "sector": "Ngành", "count": "Số mẫu",
                    "avg_return_pct": "Lợi nhuận TB (%)", "win_rate_pct": "Tỷ lệ tăng (%)",
                })
                st.dataframe(sector_paper_df, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.warning(f"Nhật ký paper trading chưa đọc được: {exc}")
else:
    st.info(
        "Chưa có nhật ký paper trading. File sẽ được tạo trong lần quét tự động kế tiếp; "
        "các chỉ tiêu T+3/T+5/T+10/T+20 sẽ xuất hiện dần khi đủ số phiên thực tế."
    )

st.divider()


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
        vnindex_for_screen = fetch_vnindex(500)  # lấy 1 lần, dùng chung cho mọi mã trong danh mục
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

                avg_trade_value_bn = float((raw["close"] * raw["volume"]).tail(20).mean()) / 1e9
                if avg_trade_value_bn < min_trade_value:
                    st.caption(f"Bỏ qua {tk}: thanh khoản {avg_trade_value_bn:.2f} tỷ đ/phiên, dưới ngưỡng lọc")
                    continue

                # Đếm nhanh số lớp đồng thuận (không tính lớp tin tức để giữ tốc độ)
                feats_sig = add_signal_columns(feats)
                latest_vol_spike = bool(feats_sig["vol_spike"].iloc[-1])
                rel_strength_tk = None
                if vnindex_for_screen is not None:
                    rel_df_tk = compute_relative_strength(raw, vnindex_for_screen)
                    if not rel_df_tk.dropna().empty:
                        rel_strength_tk = float(rel_df_tk["rel_strength"].dropna().iloc[-1])

                votes = 0
                if r1["probability"] >= 0.55:
                    votes += 1
                if rel_strength_tk is not None and rel_strength_tk > 0.02:
                    votes += 1

                rows.append({
                    "Mã": tk, "Giá": f"{last_price:,.0f}", "% ngày": f"{chg:+.2f}%",
                    "Xác suất tăng T+1": r1["probability"],
                    "Xác suất tăng T+3": r3["probability"],
                    "Acc nhanh T+1": r1["quick_accuracy"],
                    "Khối lượng bất thường": "🟡 Có" if latest_vol_spike else "⚪ Không",
                    "GTGD TB/phiên": f"{avg_trade_value_bn:,.1f} tỷ",
                    "Tín hiệu đồng thuận (tối đa 2)": votes,
                })
            except Exception as e:
                st.caption(f"Bỏ qua {tk}: {e}")
                continue
        progress.empty()

        if rows:
            rank_df = pd.DataFrame(rows).sort_values(
                ["Tín hiệu đồng thuận (tối đa 2)", "Xác suất tăng T+1"], ascending=False
            )
            display_df = rank_df.copy()
            display_df["Xác suất tăng T+1"] = display_df["Xác suất tăng T+1"].apply(lambda x: f"{x*100:.0f}%")
            display_df["Xác suất tăng T+3"] = display_df["Xác suất tăng T+3"].apply(lambda x: f"{x*100:.0f}%")
            display_df["Acc nhanh T+1"] = display_df["Acc nhanh T+1"].apply(lambda x: f"{x*100:.0f}%")
            st.dataframe(display_df, hide_index=True, use_container_width=True)
            st.caption(
                "Cột 'Tín hiệu đồng thuận' đếm nhanh 2 lớp (mô hình + sức mạnh tương đối vs VN-Index), "
                "chưa gồm tin tức để giữ tốc độ. Xem chi tiết đầy đủ 4 lớp ở phần 'Chạy dự báo chi tiết' bên dưới. "
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
    # BỐI CẢNH THỊ TRƯỜNG CHUNG (VN-Index) — đọc trước khi xem tín hiệu riêng của mã
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("🌐 Bối cảnh thị trường chung")

    with st.spinner("Đang lấy VN-Index..."):
        vnindex_df = fetch_vnindex(days_back)

    market_ctx_detail = analyze_market_context(vnindex_df)
    if market_ctx_detail:
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Xu hướng VN-Index", f"{market_ctx_detail['trend_icon']} {market_ctx_detail['trend']}")
        bc2.metric("Mức biến động", f"{market_ctx_detail['volatility_icon']} {market_ctx_detail['volatility_level']}")
        bc3.metric(
            "VN-Index 5 phiên",
            f"{market_ctx_detail['change_5d_pct']:+.2f}%" if market_ctx_detail['change_5d_pct'] is not None else "N/A",
        )
        st.caption(f"💡 {context_advisory_note(market_ctx_detail)}")
    else:
        st.caption("Không lấy được bối cảnh VN-Index.")

    # -----------------------------------------------------------------
    # TÍN HIỆU PHÁT HIỆN SỚM — tổng hợp 4 lớp độc lập
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("🔎 Tín hiệu phát hiện sớm (tổng hợp nhiều lớp)")
    st.caption(
        "Đây KHÔNG phải khuyến nghị mua/bán. Chỉ là tổng hợp nhiều góc nhìn độc lập "
        "để bạn tự đánh giá — công cụ tham khảo cá nhân, không thay thế phân tích "
        "của người có chứng chỉ hành nghề chứng khoán."
    )

    feats_sig = add_signal_columns(feats)
    latest_sig_row = feats_sig.iloc[-1]

    latest_rel_strength = None
    if vnindex_df is not None and not vnindex_df.empty:
        rel_df = compute_relative_strength(raw, vnindex_df)
        if not rel_df.dropna().empty:
            latest_rel_strength = float(rel_df["rel_strength"].dropna().iloc[-1])

    # Tái sử dụng tin tức đã lấy (tính trước ở đây, phần hiển thị đầy đủ ở dưới)
    try:
        news_df_for_signal = fetch_news_cached()
        ticker_news_for_signal = get_news_for_ticker(ticker, news_df_for_signal)
        news_summary_for_signal = summarize_sentiment(ticker_news_for_signal)
    except Exception:
        news_summary_for_signal = {"n_news": 0, "overall": "Chưa có tin"}

    combo = composite_signal(
        ml_prob_1=float(prob1),
        latest_vol_spike=bool(latest_sig_row["vol_spike"]),
        latest_rel_strength=latest_rel_strength,
        news_summary=news_summary_for_signal,
    )

    sig_cols = st.columns(4)
    for (label, value), col in zip(combo["details"].items(), sig_cols):
        with col:
            st.markdown(f"**{label}**")
            st.markdown(value)

    st.markdown(f"### {combo['verdict']}")
    st.caption(f"Phiếu đồng thuận: {combo['votes_bull']} tăng · {combo['votes_bear']} giảm (trên tối đa 3 lớp có phiếu)")

    with st.expander("📈 Kiểm định tín hiệu quy tắc bằng dữ liệu lịch sử (event-study)"):
        st.caption(
            "Tín hiệu quy tắc: RSI cắt lên/xuống mốc 50 + khối lượng đột biến (>1.5x TB20) "
            "+ xu hướng MA ủng hộ. Đây là tín hiệu ĐƠN GIẢN HƠN mô hình ML ở trên, nhưng "
            "kiểm định được trên toàn bộ lịch sử để biết có thật sự có 'edge' hay không."
        )
        es_bull = event_study(feats_sig, "signal_bull", "fut_ret_1")
        es_bear = event_study(feats_sig, "signal_bear", "fut_ret_1")

        for name, es in [("Tín hiệu TĂNG (rule-based)", es_bull), ("Tín hiệu GIẢM (rule-based)", es_bear)]:
            st.markdown(f"**{name}**")
            if es.get("insufficient"):
                st.write(f"Chỉ có {es['n_events']} lần xuất hiện trong lịch sử — chưa đủ để kết luận đáng tin cậy.")
            else:
                st.write(
                    f"Xuất hiện {es['n_events']} lần trong quá khứ. Khi tín hiệu này xảy ra, "
                    f"lợi nhuận T+1 trung bình là **{es['event_mean_return']*100:+.2f}%** "
                    f"(so với baseline mọi phiên: {es['baseline_mean_return']*100:+.2f}%) — "
                    f"chênh lệch **{es['edge_return']*100:+.2f}pp**. "
                    f"Tỷ lệ tăng giá sau tín hiệu: {es['event_pct_positive']*100:.0f}% "
                    f"(baseline: {es['baseline_pct_positive']*100:.0f}%)."
                )
                if abs(es['edge_return']) < 0.002:
                    st.caption("⚠️ Chênh lệch quá nhỏ — tín hiệu này có thể không có edge thật sự với mã này.")

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

    # -----------------------------------------------------------------
    # KẾ HOẠCH GIAO DỊCH VÀ QUẢN TRỊ VỐN
    # -----------------------------------------------------------------
    st.divider()
    st.subheader("🛡️ Kế hoạch giao dịch tham khảo")
    st.caption(
        "Vùng mua dựa trên EMA20/hỗ trợ hoặc breakout; stop là điểm vô hiệu kỹ thuật. "
        "App tự cảnh báo mua đuổi và dùng tỷ lệ rủi ro:lợi nhuận 1:1 tại TP1, 1:2 tại TP2."
    )

    detail_sector_score = None
    try:
        with open("signals_latest.json", "r", encoding="utf-8") as score_file:
            saved_scan = _json.load(score_file)
        detail_sector = sector_for_ticker(ticker)
        detail_sector_score = next(
            (row["score"] for row in saved_scan.get("sector_rankings", [])
             if row.get("sector") == detail_sector),
            None,
        )
    except Exception:
        detail_sector = sector_for_ticker(ticker)

    plan = build_trade_plan(
        raw,
        sector_score=detail_sector_score,
        market_trend=market_ctx_detail.get("trend") if market_ctx_detail else None,
    )

    if plan:
        action_colors = {
            "MUA THĂM DÒ": "green", "KHÔNG MUA": "red",
            "KHÔNG MUA ĐUỔI": "orange", "CHỜ XÁC NHẬN": "orange",
        }
        color = action_colors.get(plan["action"], "blue")
        st.markdown(f"### :{color}[{plan['action']}]")
        st.write(f"**Thiết lập:** {plan['setup']} · {plan['reason']}")
        if detail_sector_score is not None:
            st.caption(f"Ngành {detail_sector}: {detail_sector_score:.1f}/100")

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Vùng mua thấp", f"{plan['entry_low']:,.0f} đ")
        pc2.metric("Vùng mua cao", f"{plan['entry_high']:,.0f} đ")
        pc3.metric("Cắt lỗ", f"{plan['stop_loss']:,.0f} đ", f"-{plan['risk_pct']:.1f}% từ giá mua chuẩn")
        pc4.metric("ATR hiện tại", f"{plan['atr_pct']:.1f}%")

        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("TP1 – chốt 30–40%", f"{plan['tp1']:,.0f} đ", "+1R")
        tc2.metric("TP2 – chốt thêm", f"{plan['tp2']:,.0f} đ", "+2R")
        tc3.metric("Trailing stop tham khảo", f"{plan['trailing_stop']:,.0f} đ")

        with st.expander("Tính số lượng mua theo tổng vốn"):
            capital = st.number_input(
                "Tổng vốn tài khoản (đồng)", min_value=10_000_000,
                value=100_000_000, step=10_000_000,
            )
            risk_percent = st.select_slider(
                "Mức rủi ro tối đa mỗi lệnh", options=[0.5, 0.75, 1.0, 1.5, 2.0], value=1.0,
                format_func=lambda value: f"{value}%",
            )
            max_position_percent = st.slider(
                "Tỷ trọng tối đa cho một mã", 5, 30, 15, step=5,
            )
            position = calculate_position_size(
                capital=float(capital), entry_price=plan["entry_reference"],
                stop_loss=plan["stop_loss"], risk_percent=float(risk_percent),
                max_position_percent=float(max_position_percent),
            )
            if position["quantity"] > 0:
                st.metric("Khối lượng tối đa tham khảo", f"{position['quantity']:,} cổ phiếu")
                st.caption(
                    f"Giá trị vị thế khoảng {position['position_value']:,.0f}đ · "
                    f"Vốn chịu rủi ro nếu chạm stop khoảng {position['capital_at_risk']:,.0f}đ · "
                    f"Giới hạn bởi: {position['limited_by']}."
                )
            else:
                st.warning("Quy mô vốn/rủi ro hiện tại chưa đủ mua một lô 100 cổ phiếu trong giới hạn đã chọn.")

        st.caption(
            f"Cơ sở stop: {plan['stop_basis']}. Nếu luận điểm ngành hoặc thị trường xấu đi, "
            "cần đánh giá thoát sớm thay vì chờ máy móc đến đúng mức stop."
        )
    else:
        st.info("Không đủ dữ liệu để lập kế hoạch giao dịch cho mã này.")

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
    st.subheader("📰 Tin tức liên quan đã phân loại")
    st.caption(
        "Tin được loại trùng, nhận diện mã/ngành, phân loại sự kiện và chấm mức tác động. "
        "Điểm tin tức được dùng cho 10% điểm ngành, nhưng chưa đưa trực tiếp vào mô hình ML của từng mã. "
        "Bạn vẫn nên mở bài gốc để kiểm tra ngữ cảnh."
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
                    st.caption(
                        f"Sự kiện: {row.get('event_type', 'Tin chung')} · "
                        f"Mức tác động: {row.get('impact_level', 'Chưa rõ')}"
                    )
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
