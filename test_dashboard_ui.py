from dashboard_ui import prepare_horizon_table, prepare_sector_table, prepare_stock_table


def test_old_stock_schema_remains_readable():
    table = prepare_stock_table([{"ticker": "FPT", "price": 100, "change_pct": 1.2}])
    assert table.iloc[0]["Mã"] == "FPT"
    assert table.iloc[0]["Chất lượng"] == "Chưa kiểm định"
    assert table.iloc[0]["Vùng mua"] == "N/A"


def test_new_stock_schema_is_flattened():
    row = {"ticker": "FPT", "price": 100, "change_pct": 1, "sector": "Công nghệ",
           "model_quality": {"label": "🟢 Đạt", "model_edge_pp": 3},
           "trade_plan": {"action": "MUA THĂM DÒ", "entry_low": 98, "entry_high": 101,
                          "stop_loss": 94, "tp2": 112}}
    table = prepare_stock_table([row])
    assert table.iloc[0]["Trạng thái"] == "MUA THĂM DÒ"
    assert table.iloc[0]["Vùng mua"] == "98–101"


def test_old_sector_schema_gets_news_defaults():
    row = {"sector": "Công nghệ", "score": 70, "status": "Đang cải thiện",
           "relative_strength_20d_pct": 3, "breadth_ma20_pct": 80}
    table = prepare_sector_table([row])
    assert table.iloc[0]["Điểm tin"] == "Chưa có"
    assert table.iloc[0]["Bằng chứng"] == "0 tin"


def test_horizon_table_always_has_four_rows():
    table = prepare_horizon_table({"horizons": {"t5": {"count": 2, "win_rate_pct": 50, "avg_return_pct": 1}}})
    assert table["Mốc"].tolist() == ["T+3", "T+5", "T+10", "T+20"]
