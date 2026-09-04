"""Huấn luyện, kiểm định theo thời gian và dự báo vùng giá."""

import numpy as np
import pandas as pd

from features import FEATURE_COLS

TARGET_HORIZONS = {"target_1": 1, "target_3": 3}


def target_horizon(target_col: str) -> int:
    if target_col not in TARGET_HORIZONS:
        raise ValueError(f"Target không được hỗ trợ: {target_col}")
    return TARGET_HORIZONS[target_col]


def purged_train_end(test_start: int, horizon: int) -> int:
    """Loại mẫu train có nhãn nhìn sang giai đoạn test."""
    return max(0, test_start - horizon)


def _classifier():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, verbose=-1, random_state=42,
    )


def walk_forward_eval(d: pd.DataFrame, target_col: str, min_train: int = 150):
    """Walk-forward có purge theo kỳ dự báo để tránh rò rỉ dữ liệu."""
    horizon = target_horizon(target_col)
    data = d.dropna(subset=FEATURE_COLS + [target_col]).reset_index(drop=True)
    if len(data) < min_train + 20:
        return None

    preds, probabilities, actuals, baseline_preds = [], [], [], []
    for test_start in range(min_train + horizon, len(data), 5):
        train = data.iloc[:purged_train_end(test_start, horizon)]
        test = data.iloc[test_start:test_start + 5]
        if len(train) < min_train or test.empty or train[target_col].nunique() < 2:
            continue
        model = _classifier()
        model.fit(train[FEATURE_COLS], train[target_col].astype(int))
        proba = model.predict_proba(test[FEATURE_COLS])[:, 1]
        probabilities.extend(proba.tolist())
        preds.extend((proba >= 0.5).astype(int).tolist())
        actuals.extend(test[target_col].astype(int).tolist())
        baseline_preds.extend((test["ret_1"] > 0).astype(int).tolist())

    if not preds:
        return None
    pred_arr, actual_arr = np.asarray(preds), np.asarray(actuals)
    proba_arr = np.asarray(probabilities)
    final_model = _classifier()
    final_model.fit(data[FEATURE_COLS], data[target_col].astype(int))
    return {
        "accuracy": float(np.mean(pred_arr == actual_arr)),
        "baseline_accuracy": float(np.mean(np.asarray(baseline_preds) == actual_arr)),
        "brier_score": float(np.mean((proba_arr - actual_arr) ** 2)),
        "n_test": len(preds), "horizon": horizon, "model": final_model,
    }


def confidence_label(acc: float, baseline: float) -> str:
    edge = acc - baseline
    if edge >= 0.06:
        return "cao"
    if edge >= 0.02:
        return "trung bình"
    return "thấp — gần bằng hoặc kém baseline"


def train_quantile_models(d: pd.DataFrame, ret_col: str, min_train: int = 150):
    from lightgbm import LGBMRegressor
    data = d.dropna(subset=FEATURE_COLS + [ret_col]).reset_index(drop=True)
    if len(data) < min_train:
        return None
    models = {}
    for q in [0.1, 0.5, 0.9]:
        model = LGBMRegressor(
            objective="quantile", alpha=q, n_estimators=80, max_depth=3,
            learning_rate=0.08, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=42,
        )
        model.fit(data[FEATURE_COLS], data[ret_col])
        models[q] = model
    return models


def predict_price_range(models: dict, latest_row: pd.DataFrame, current_price: float):
    preds = {q: float(m.predict(latest_row[FEATURE_COLS])[0]) for q, m in models.items()}
    lo_ret, mid_ret, hi_ret = sorted(preds.values())
    return {
        "lo_price": current_price * (1 + lo_ret), "mid_price": current_price * (1 + mid_ret),
        "hi_price": current_price * (1 + hi_ret), "lo_pct": lo_ret * 100,
        "mid_pct": mid_ret * 100, "hi_pct": hi_ret * 100,
    }


def quick_train_predict(d: pd.DataFrame, target_col: str, test_frac: float = 0.2):
    """Holdout theo thời gian có purge; dự báo dòng feature mới nhất chưa gắn nhãn."""
    horizon = target_horizon(target_col)
    data = d.dropna(subset=FEATURE_COLS + [target_col]).reset_index(drop=True)
    if len(data) < 100:
        return None
    split = int(len(data) * (1 - test_frac))
    train, test = data.iloc[:purged_train_end(split, horizon)], data.iloc[split:]
    if len(test) < 10 or train[target_col].nunique() < 2:
        return None
    model = _classifier()
    model.fit(train[FEATURE_COLS], train[target_col].astype(int))
    test_proba = model.predict_proba(test[FEATURE_COLS])[:, 1]
    actual = test[target_col].astype(int).to_numpy()
    final_model = _classifier()
    final_model.fit(data[FEATURE_COLS], data[target_col].astype(int))
    latest_row = d.dropna(subset=FEATURE_COLS).iloc[[-1]]
    return {
        "probability": float(final_model.predict_proba(latest_row[FEATURE_COLS])[0][1]),
        "quick_accuracy": float(np.mean((test_proba >= 0.5).astype(int) == actual)),
        "brier_score": float(np.mean((test_proba - actual) ** 2)),
        "n_test": len(test), "horizon": horizon,
    }
