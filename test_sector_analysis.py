import numpy as np
import pandas as pd

from sector_analysis import attach_score_changes, calculate_sector_rankings, classify_sector


def _history(total_return, volume_ratio=1.0, n=80):
    close = 100 * np.exp(np.linspace(0, np.log(1 + total_return), n))
    volume = np.full(n, 1_000_000.0)
    volume[-1] *= volume_ratio
    return pd.DataFrame({"close": close, "volume": volume})


def test_stronger_sector_ranks_above_weaker_sector():
    histories = {
        "FPT": _history(0.30, 1.8),
        "ACB": _history(-0.12, 0.7),
        "VCB": _history(-0.08, 0.8),
    }
    result = calculate_sector_rankings(histories, _history(0.02))
    assert result[0]["sector"] == "Công nghệ"
    assert result[0]["score"] > result[-1]["score"]
    assert 0 <= result[0]["score"] <= 100


def test_score_change_only_when_previous_value_exists():
    current = [{"sector": "Ngân hàng", "score": 65.0}, {"sector": "Thép", "score": 50.0}]
    result = attach_score_changes(current, [{"sector": "Ngân hàng", "score": 60.0}])
    assert result[0]["score_change"] == 5.0
    assert result[1]["score_change"] is None


def test_sector_status_boundaries():
    assert classify_sector(75) == "Dẫn dắt mạnh"
    assert classify_sector(60) == "Đang cải thiện"
    assert classify_sector(45) == "Trung tính / tích lũy"
    assert classify_sector(30) == "Suy yếu"
    assert classify_sector(29.9) == "Điều chỉnh mạnh"
