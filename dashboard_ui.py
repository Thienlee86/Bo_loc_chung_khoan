"""Dashboard nhiều tab, ưu tiên hiển thị gọn trên điện thoại."""

from __future__ import annotations

import pandas as pd


def prepare_sector_table(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).copy()
    defaults = {"score_change": None, "news_score": None, "news_count": 0}
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
    icons = {"Dẫn dắt mạnh": "🟢", "Đang cải thiện": "🔵", "Trung tính / tích lũy": "⚪",
             "Suy yếu": "🟡", "Điều chỉnh mạnh": "🔴"}
    frame["status"] = frame["status"].map(lambda value: f"{icons.get(value, '⚪')} {value}")
    frame["score_change"] = frame["score_change"].map(lambda v: "Mới" if pd.isna(v) else f"{v:+.1f}")
    frame["relative_strength_20d_pct"] = frame["relative_strength_20d_pct"].map(lambda v: f"{v:+.1f}%")
    frame["breadth_ma20_pct"] = frame["breadth_ma20_pct"].map(lambda v: f"{v:.0f}%")
    frame["news_score"] = frame["news_score"].map(lambda v: "Chưa có" if pd.isna(v) else f"{v:.0f}/100")
    frame["news_count"] = frame["news_count"].map(lambda v: f"{int(v)} tin")
    columns = ["sector", "score", "score_change", "status", "relative_strength_20d_pct",
               "breadth_ma20_pct", "news_score", "news_count"]
    return frame[columns].rename(columns={
        "sector": "Nhóm ngành", "score": "Điểm", "score_change": "Thay đổi",
        "status": "Trạng thái", "relative_strength_20d_pct": "Mạnh/yếu 20P",
        "breadth_ma20_pct": "% mã trên MA20", "news_score": "Điểm tin", "news_count": "Bằng chứng",
    })


def prepare_stock_table(rows: list[dict]) -> pd.DataFrame:
    """Làm phẳng dữ liệu mới và vẫn đọc được schema cũ thiếu plan/quality."""
    if not rows:
        return pd.DataFrame()
    output = []
    for row in rows:
        plan = row.get("trade_plan") if isinstance(row.get("trade_plan"), dict) else {}
        quality = row.get("model_quality") if isinstance(row.get("model_quality"), dict) else {}
        output.append({
            "Mã": row.get("ticker"), "Ngành": row.get("sector", "Chưa có"),
            "Giá": row.get("price"), "% ngày": f"{row.get('change_pct', 0):+.2f}%",
            "Điểm ngành": row.get("sector_score"),
            "Chất lượng": quality.get("label", "Chưa kiểm định"),
            "Lợi thế": "N/A" if quality.get("model_edge_pp") is None else f"{quality['model_edge_pp']:+.1f} điểm %",
            "Trạng thái": plan.get("action", "Chưa có"),
            "Vùng mua": "N/A" if not plan else f"{plan['entry_low']:,.0f}–{plan['entry_high']:,.0f}",
            "Cắt lỗ": "N/A" if not plan else f"{plan['stop_loss']:,.0f}",
            "TP2": "N/A" if not plan else f"{plan['tp2']:,.0f}",
        })
    return pd.DataFrame(output)


def prepare_horizon_table(summary: dict) -> pd.DataFrame:
    rows = []
    horizons = summary.get("horizons", {})
    for key, label in [("t3", "T+3"), ("t5", "T+5"), ("t10", "T+10"), ("t20", "T+20")]:
        metric = horizons.get(key, {})
        rows.append({
            "Mốc": label, "Số mẫu": metric.get("count", 0),
            "Tỷ lệ tăng": "Chưa đủ" if metric.get("win_rate_pct") is None else f"{metric['win_rate_pct']:.1f}%",
            "Lợi nhuận ròng TB": "Chưa đủ" if metric.get("avg_return_pct") is None else f"{metric['avg_return_pct']:+.2f}%",
        })
    return pd.DataFrame(rows)


def _render_overview(st, scan_data: dict, market_ctx: dict | None):
    st.subheader("Tổng quan hôm nay")
    if market_ctx:
        c1, c2, c3 = st.columns(3)
        c1.metric("VN-Index", market_ctx.get("trend", "N/A"))
        c2.metric("Biến động", market_ctx.get("volatility_level", "N/A"))
        change = market_ctx.get("change_5d_pct")
        c3.metric("Thay đổi 5 phiên", "N/A" if change is None else f"{change:+.2f}%")
    health = scan_data.get("model_health", {})
    if health:
        st.markdown(f"**Sức khỏe mô hình: {health.get('status', 'Chưa có')}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Đạt", health.get("pass", 0)); c2.metric("Chờ", health.get("caution", 0))
        c3.metric("Không đạt", health.get("block", 0))
        edge = health.get("avg_edge_pp")
        c4.metric("Lợi thế TB", "N/A" if edge is None else f"{edge:+.1f} điểm %")
    sectors = scan_data.get("sector_rankings", [])[:3]
    if sectors:
        st.markdown("**Ba nhóm ngành có điểm cao nhất**")
        cols = st.columns(len(sectors))
        for col, row in zip(cols, sectors):
            col.metric(row["sector"], f"{row['score']:.1f}/100", row.get("status"))
    if not scan_data:
        st.info("Chưa có kết quả quét. Hãy chạy workflow Quét danh mục hằng ngày.")


def _render_news(st, news_loader):
    st.subheader("Tin tức đã phân loại")
    try:
        from smart_news import enrich_news
        news = enrich_news(news_loader())
    except Exception as exc:
        st.warning(f"Chưa lấy được tin tức: {exc}"); return
    if news.empty:
        st.info("Chưa có tin tức trong các nguồn RSS hiện tại."); return
    sector_options = sorted({sector for items in news["sectors"] for sector in items})
    selected = st.selectbox("Lọc theo ngành", ["Tất cả"] + sector_options)
    if selected != "Tất cả":
        news = news[news["sectors"].map(lambda items: selected in items)]
    for _, row in news.head(20).iterrows():
        icon = {"Tích cực": "🟢", "Tiêu cực": "🔴"}.get(row["sentiment_label"], "⚪")
        with st.expander(f"{icon} {row['title']}"):
            st.write(row.get("summary") or "Không có tóm tắt.")
            st.caption(f"{row['event_type']} · Tác động {row['impact_level']} · {row['source']}")
            if row.get("link"):
                st.link_button("Đọc bài gốc", row["link"])


def _render_paper(st, paper_data: dict):
    st.subheader("Kiểm định paper trading")
    if not paper_data:
        st.info("Nhật ký sẽ được tạo sau lần quét tự động kế tiếp."); return
    summary = paper_data.get("summary", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tổng tín hiệu", summary.get("total_signals", 0)); c2.metric("Đang mở", summary.get("open_signals", 0))
    win = summary.get("win_rate_pct"); avg = summary.get("avg_net_return_pct")
    c3.metric("Tỷ lệ thắng", "Chưa đủ" if win is None else f"{win:.1f}%")
    c4.metric("Lợi nhuận TB", "Chưa đủ" if avg is None else f"{avg:+.2f}%")
    st.dataframe(prepare_horizon_table(summary), hide_index=True, use_container_width=True)
    trades = paper_data.get("trades", [])[:20]
    if trades:
        frame = pd.DataFrame([{
            "Ngày": t.get("signal_date"), "Mã": t.get("ticker"), "Ngành": t.get("sector"),
            "Trạng thái": t.get("status"), "T+5": t.get("horizon_returns_pct", {}).get("t5"),
            "Kết quả lệnh": t.get("net_return_pct"), "Lý do thoát": t.get("exit_reason") or "—",
        } for t in trades])
        st.dataframe(frame, hide_index=True, use_container_width=True)
    st.caption("Kết quả đã trừ 0,30% chi phí giả định; stop được ưu tiên nếu stop và TP cùng chạm trong một nến ngày.")


def _render_method(st):
    st.subheader("Phương pháp và cách đọc")
    st.markdown("""
1. **Thị trường:** kiểm tra xu hướng và biến động VN-Index.
2. **Ngành:** điểm 0–100 từ sức mạnh tương đối, độ rộng, dòng tiền, động lượng, mã dẫn dắt và tin tức.
3. **Cổ phiếu:** mô hình phải vượt baseline; Brier Score dùng để kiểm tra chất lượng xác suất.
4. **Kế hoạch:** chỉ mua trong vùng hợp lệ, stop tại điểm vô hiệu, TP1 = 1R và TP2 = 2R.
5. **Kiểm định:** tín hiệu được lưu trước, sau đó mới đối chiếu T+3/T+5/T+10/T+20.

Màu vàng thường có nghĩa là **chưa đủ bằng chứng**, không đồng nghĩa tín hiệu xấu. Công cụ này hỗ trợ quản trị quyết định, không bảo đảm lợi nhuận.
""")


def render_dashboard(scan_data: dict, paper_data: dict, market_ctx: dict | None, news_loader):
    import streamlit as st
    tabs = st.tabs(["Tổng quan", "Nhóm ngành", "Cổ phiếu", "Tin tức", "Paper", "Phương pháp"])
    with tabs[0]: _render_overview(st, scan_data, market_ctx)
    with tabs[1]:
        st.subheader("Bản đồ sức mạnh nhóm ngành")
        table = prepare_sector_table(scan_data.get("sector_rankings", []))
        st.dataframe(table, hide_index=True, use_container_width=True) if not table.empty else st.info("Chưa có dữ liệu ngành.")
    with tabs[2]:
        st.subheader("Cổ phiếu trong danh mục")
        table = prepare_stock_table(scan_data.get("results", []))
        st.dataframe(table, hide_index=True, use_container_width=True) if not table.empty else st.info("Chưa có dữ liệu cổ phiếu.")
    with tabs[3]: _render_news(st, news_loader)
    with tabs[4]: _render_paper(st, paper_data)
    with tabs[5]: _render_method(st)
