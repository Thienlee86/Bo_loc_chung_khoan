"""
Tầng nghiệp vụ (business logic) — huấn luyện mô hình, kiểm định walk-forward,
dự báo vùng giá bằng quantile regression, và xếp hạng nhiều mã.

Tách riêng khỏi app.py theo kiến trúc phân tầng.
"""

import numpy as np
import pandas as pd

from features import FEATURE_COLS


def walk_forward_eval(d: pd.DataFrame, target_col: str, min_train: int = 150):
    """Đánh giá walk-forward: train trên quá khứ, dự báo từng điểm tương lai kế tiếp.
    Trả về accuracy mô hình, accuracy baseline, và model cuối cùng train trên toàn bộ dữ liệu."""
    from lightgbm import LGBMClassifier

    data = d.dropna(subset=FEATURE_COLS + [target_col]).reset_index(drop=True)
    if len(data) < min_train + 20:
        return None

    preds, actuals, baseline_preds = [], [], []
    step = 5

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
        baseline_preds.extend((test["ret_1"] > 0).astype(int).tolist())

    if not preds:
        return None

    acc = float(np.mean(np.array(preds) == np.array(actuals)))
    baseline_acc = float(np.mean(np.array(baseline_preds) == np.array(actuals)))

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
# VÙNG GIÁ DỰ BÁO (quantile regression)
# ---------------------------------------------------------------------------

def train_quantile_models(d: pd.DataFrame, ret_col: str, min_train: int = 150):
    """Train 3 mô hình hồi quy phân vị (10%, 50%, 90%) dự báo % thay đổi giá.
    Trả về dict 3 model, hoặc None nếu không đủ dữ liệu.

    LƯU Ý: khoảng giá ra được phản ánh ĐÚNG mức biến động thực tế của mã —
    có thể rộng hơn kỳ vọng, đây không phải lỗi mà là bản chất của dự báo ngắn hạn."""
    from lightgbm import LGBMRegressor

    data = d.dropna(subset=FEATURE_COLS + [ret_col]).reset_index(drop=True)
    if len(data) < min_train:
        return None

    models = {}
    for q in [0.1, 0.5, 0.9]:
        m = LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=80, max_depth=3, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, verbose=-1,
        )
        m.fit(data[FEATURE_COLS], data[ret_col])
        models[q] = m

    return models


def predict_price_range(models: dict, latest_row: pd.DataFrame, current_price: float):
    """Dùng 3 model quantile để dự báo vùng giá. Đảm bảo thứ tự lo <= mid <= hi
    (các model train riêng lẻ đôi khi cho kết quả bắt chéo nhau, cần sắp lại)."""
    preds = {q: float(m.predict(latest_row[FEATURE_COLS])[0]) for q, m in models.items()}
    rets_sorted = sorted(preds.values())
    lo_ret, mid_ret, hi_ret = rets_sorted

    return {
        "lo_price": current_price * (1 + lo_ret),
        "mid_price": current_price * (1 + mid_ret),
        "hi_price": current_price * (1 + hi_ret),
        "lo_pct": lo_ret * 100, "mid_pct": mid_ret * 100, "hi_pct": hi_ret * 100,
    }


# ---------------------------------------------------------------------------
# XẾP HẠNG NHIỀU MÃ (screening)
# ---------------------------------------------------------------------------

def quick_train_predict(d: pd.DataFrame, target_col: str, test_frac: float = 0.2):
    """Phiên bản NHANH (không walk-forward đầy đủ) dùng cho bảng xếp hạng nhiều mã —
    chỉ 1 lần chia train/test theo thời gian để có accuracy tham khảo nhanh, KHÔNG
    thay thế cho kiểm định walk-forward đầy đủ ở phần xem chi tiết 1 mã.

    Trả về: xác suất dự báo mới nhất, accuracy trên tập test giữ lại cuối chuỗi."""
    from lightgbm import LGBMClassifier

    data = d.dropna(subset=FEATURE_COLS + [target_col]).reset_index(drop=True)
    if len(data) < 100:
        return None

    split = int(len(data) * (1 - test_frac))
    train, test = data.iloc[:split], data.iloc[split:-1]  # bỏ dòng cuối vì target là NaN thực tế đã dropna

    if len(test) < 10 or train[target_col].nunique() < 2:
        return None

    model = LGBMClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
    )
    model.fit(train[FEATURE_COLS], train[target_col])

    test_pred = model.predict(test[FEATURE_COLS])
    acc = float(np.mean(test_pred == test[target_col].values))

    # Train lại trên toàn bộ dữ liệu sạch để dự báo phiên mới nhất
    final_model = LGBMClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, verbose=-1,
    )
    final_model.fit(data[FEATURE_COLS], data[target_col])

    latest_row = d.dropna(subset=FEATURE_COLS).iloc[[-1]]
    prob = float(final_model.predict_proba(latest_row[FEATURE_COLS])[0][1])

    return {"probability": prob, "quick_accuracy": acc, "n_test": len(test)}
