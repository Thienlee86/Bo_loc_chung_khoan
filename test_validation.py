import numpy as np
import pandas as pd

from features import _future_direction, build_features
from models import purged_train_end, target_horizon


def _sample_ohlcv(n=80):
    close = pd.Series(np.linspace(10, 20, n))
    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="B"),
        "open": close * 0.995, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": np.linspace(1_000_000, 2_000_000, n),
    })


def test_unknown_future_labels_remain_nan():
    result = build_features(_sample_ohlcv())
    assert result["target_1"].tail(1).isna().all()
    assert result["target_3"].tail(3).isna().all()
    assert result["fut_ret_1"].tail(1).isna().all()
    assert result["fut_ret_3"].tail(3).isna().all()


def test_known_direction_is_encoded_correctly():
    target = _future_direction(pd.Series([10.0, 11.0, 9.0, 12.0]), 1)
    assert target.iloc[:3].tolist() == [1.0, 0.0, 1.0]
    assert np.isnan(target.iloc[-1])


def test_purge_matches_prediction_horizon():
    assert target_horizon("target_1") == 1
    assert target_horizon("target_3") == 3
    assert purged_train_end(100, 1) == 99
    assert purged_train_end(100, 3) == 97
