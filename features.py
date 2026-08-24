"""
Tầng chuẩn hoá dữ liệu (feature engineering) — tính các chỉ báo kỹ thuật
từ dữ liệu giá thô (OHLCV) thành các đặc trưng đưa vào mô hình.

Tách riêng khỏi app.py theo kiến trúc phân tầng: raw data -> features -> model -> UI.
"""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toàn bộ chỉ báo kỹ thuật từ DataFrame giá thô (cột: date, open, high, low, close, volume)."""
    d = df.copy()

    # --- Đường trung bình & giao cắt ---
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma_cross"] = (d["ma5"] - d["ma20"]) / d["ma20"]

    # --- RSI(14) ---
    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))

    # --- MACD ---
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # --- Volume ---
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()

    # --- True Range & ATR ---
    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean() / d["close"]

    # --- Bollinger %B ---
    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    d["boll_pctb"] = (d["close"] - lower) / (upper - lower)

    # --- Lợi nhuận gần đây (momentum) ---
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_3"] = d["close"].pct_change(3)

    # --- ADX(14): độ mạnh xu hướng ---
    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).rolling(14).sum() / tr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(14).sum() / tr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["adx14"] = dx.rolling(14).mean() / 100  # chuẩn hoá về [0,1] cho gần thang các feature khác

    # --- OBV (On-Balance Volume), chuẩn hoá theo rolling z-score để ổn định thang đo ---
    obv_raw = (np.sign(d["close"].diff().fillna(0)) * d["volume"]).cumsum()
    obv_mean = obv_raw.rolling(20).mean()
    obv_std = obv_raw.rolling(20).std()
    d["obv_z"] = (obv_raw - obv_mean) / obv_std.replace(0, np.nan)

    # --- Stochastic %K / %D (14, 3) ---
    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    stoch_k = 100 * (d["close"] - low14) / (high14 - low14).replace(0, np.nan)
    d["stoch_k"] = stoch_k / 100
    d["stoch_d"] = stoch_k.rolling(3).mean() / 100

    # --- Nhãn phân loại: tăng/giảm sau 1 phiên và sau 3 phiên ---
    d["target_1"] = (d["close"].shift(-1) > d["close"]).astype(int)
    d["target_3"] = (d["close"].shift(-3) > d["close"]).astype(int)

    # --- Nhãn hồi quy: % thay đổi giá tương lai (dùng cho vùng giá quantile) ---
    d["fut_ret_1"] = d["close"].shift(-1) / d["close"] - 1
    d["fut_ret_3"] = d["close"].shift(-3) / d["close"] - 1

    return d


FEATURE_COLS = [
    "ma_cross", "rsi14", "macd_hist", "vol_ratio", "atr14", "boll_pctb",
    "ret_1", "ret_3", "adx14", "obv_z", "stoch_k", "stoch_d",
]

FEATURE_LABELS = {
    "ma_cross": "Giao cắt MA5/MA20", "rsi14": "RSI(14)", "macd_hist": "MACD histogram",
    "vol_ratio": "Khối lượng / TB20", "atr14": "Biến động (ATR)", "boll_pctb": "Bollinger %B",
    "ret_1": "Lợi nhuận phiên trước", "ret_3": "Lợi nhuận 3 phiên trước",
    "adx14": "ADX(14) - độ mạnh xu hướng", "obv_z": "OBV (dòng tiền tích luỹ, chuẩn hoá)",
    "stoch_k": "Stochastic %K", "stoch_d": "Stochastic %D",
}
