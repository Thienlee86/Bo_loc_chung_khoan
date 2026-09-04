"""Feature engineering cho dữ liệu OHLCV."""

import numpy as np
import pandas as pd


def _future_direction(close: pd.Series, horizon: int) -> pd.Series:
    """Tạo nhãn hướng giá, giữ NaN khi chưa có kết quả tương lai."""
    future_close = close.shift(-horizon)
    target = pd.Series(np.nan, index=close.index, dtype="float64")
    known = future_close.notna() & close.notna()
    target.loc[known] = (future_close.loc[known] > close.loc[known]).astype(float)
    return target


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["ma_cross"] = (d["ma5"] - d["ma20"]) / d["ma20"]

    delta = d["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()

    tr = pd.concat([
        d["high"] - d["low"],
        (d["high"] - d["close"].shift()).abs(),
        (d["low"] - d["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean() / d["close"]

    mid = d["close"].rolling(20).mean()
    std = d["close"].rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    d["boll_pctb"] = (d["close"] - lower) / (upper - lower)
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_3"] = d["close"].pct_change(3)

    up_move = d["high"].diff()
    down_move = -d["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * pd.Series(plus_dm, index=d.index).rolling(14).sum() / tr14.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=d.index).rolling(14).sum() / tr14.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    d["adx14"] = dx.rolling(14).mean() / 100

    obv_raw = (np.sign(d["close"].diff().fillna(0)) * d["volume"]).cumsum()
    obv_mean = obv_raw.rolling(20).mean()
    obv_std = obv_raw.rolling(20).std()
    d["obv_z"] = (obv_raw - obv_mean) / obv_std.replace(0, np.nan)

    low14 = d["low"].rolling(14).min()
    high14 = d["high"].rolling(14).max()
    stoch_k = 100 * (d["close"] - low14) / (high14 - low14).replace(0, np.nan)
    d["stoch_k"] = stoch_k / 100
    d["stoch_d"] = stoch_k.rolling(3).mean() / 100

    d["target_1"] = _future_direction(d["close"], 1)
    d["target_3"] = _future_direction(d["close"], 3)
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
