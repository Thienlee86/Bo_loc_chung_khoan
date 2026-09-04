from model_monitor import evaluate_ticker_quality, summarize_model_health


def _result(edge=0.05, brier=0.22, n=40):
    return {"ticker": "FPT", "quick_accuracy": 0.55, "quick_baseline_accuracy": 0.55 - edge,
            "brier_score": brier, "n_test": n}


def _trades(values):
    return [{"ticker": "FPT", "horizon_returns_pct": {"t5": value}} for value in values]


def test_good_holdout_without_paper_stays_caution():
    report = evaluate_ticker_quality(_result(), [])
    assert report["gate"] == "CAUTION"


def test_model_below_baseline_is_blocked():
    report = evaluate_ticker_quality(_result(edge=-0.01), _trades([1] * 8))
    assert report["gate"] == "BLOCK"


def test_bad_probability_calibration_is_blocked():
    report = evaluate_ticker_quality(_result(brier=0.30), _trades([1] * 8))
    assert report["gate"] == "BLOCK"


def test_good_holdout_and_paper_can_pass():
    report = evaluate_ticker_quality(_result(), _trades([2, 1, -0.5, 1, 2, -0.5, 1, 1]))
    assert report["gate"] == "PASS"


def test_health_summary_warns_when_many_models_blocked():
    rows = [{"model_quality": {"gate": "BLOCK", "model_edge_pp": -1}},
            {"model_quality": {"gate": "BLOCK", "model_edge_pp": 0}},
            {"model_quality": {"gate": "CAUTION", "model_edge_pp": 1}}]
    assert "suy giảm" in summarize_model_health(rows)["status"]
