"""Đánh giá chất lượng mô hình từ holdout hiện tại và paper trading thực tế."""

from __future__ import annotations

import numpy as np


MIN_HOLDOUT_SAMPLES = 20
MIN_PAPER_SAMPLES = 8


def _ticker_paper_metrics(ticker: str, trades: list[dict], horizon: str = "t5") -> dict:
    values = []
    for trade in trades:
        if trade.get("ticker") != ticker:
            continue
        value = trade.get("horizon_returns_pct", {}).get(horizon)
        if value is not None:
            values.append(float(value))
    return {
        "count": len(values),
        "win_rate_pct": round(sum(v > 0 for v in values) / len(values) * 100, 1) if values else None,
        "avg_return_pct": round(float(np.mean(values)), 2) if values else None,
    }


def evaluate_ticker_quality(result: dict, trades: list[dict] | None = None) -> dict:
    """PASS chỉ khi cả kiểm định nhanh và paper T+5 đều đủ bằng chứng."""
    accuracy = result.get("quick_accuracy")
    baseline = result.get("quick_baseline_accuracy")
    brier = result.get("brier_score")
    n_test = int(result.get("n_test") or 0)
    edge = accuracy - baseline if accuracy is not None and baseline is not None else None
    paper = _ticker_paper_metrics(result.get("ticker", ""), trades or [])

    blockers, cautions = [], []
    if edge is None:
        blockers.append("Thiếu kết quả so sánh baseline")
    elif edge <= 0:
        blockers.append("Mô hình không vượt baseline")
    elif edge < 0.02:
        cautions.append("Lợi thế so với baseline dưới 2 điểm %")
    if n_test < MIN_HOLDOUT_SAMPLES:
        cautions.append(f"Holdout mới có {n_test}/{MIN_HOLDOUT_SAMPLES} mẫu")
    if brier is None:
        cautions.append("Chưa có Brier Score")
    elif brier > 0.27:
        blockers.append("Xác suất hiệu chỉnh kém (Brier > 0,27)")
    elif brier > 0.25:
        cautions.append("Brier Score chưa tốt hơn mốc 0,25")

    if paper["count"] < MIN_PAPER_SAMPLES:
        cautions.append(f"Paper T+5 mới có {paper['count']}/{MIN_PAPER_SAMPLES} mẫu")
    elif paper["avg_return_pct"] <= 0 or paper["win_rate_pct"] < 50:
        blockers.append("Paper T+5 chưa tạo lợi nhuận dương ổn định")

    if blockers:
        gate, label = "BLOCK", "🔴 Mất lợi thế / không đạt"
    elif cautions:
        gate, label = "CAUTION", "🟡 Chờ kiểm định thêm"
    else:
        gate, label = "PASS", "🟢 Đạt điều kiện"
    return {
        "ticker": result.get("ticker"), "gate": gate, "label": label,
        "model_edge_pp": round(edge * 100, 2) if edge is not None else None,
        "quick_accuracy_pct": round(accuracy * 100, 1) if accuracy is not None else None,
        "baseline_accuracy_pct": round(baseline * 100, 1) if baseline is not None else None,
        "brier_score": round(float(brier), 3) if brier is not None else None,
        "holdout_samples": n_test, "paper_t5": paper,
        "reasons": blockers + cautions,
    }


def attach_quality_reports(results: list[dict], trades: list[dict] | None = None) -> list[dict]:
    output = []
    for row in results:
        item = dict(row)
        item["model_quality"] = evaluate_ticker_quality(item, trades)
        output.append(item)
    return output


def summarize_model_health(results: list[dict]) -> dict:
    reports = [row.get("model_quality", {}) for row in results]
    counts = {gate: sum(r.get("gate") == gate for r in reports) for gate in ("PASS", "CAUTION", "BLOCK")}
    edges = [r.get("model_edge_pp") for r in reports if r.get("model_edge_pp") is not None]
    if counts["BLOCK"] > max(1, len(reports) * 0.3):
        status = "🔴 Chất lượng suy giảm"
    elif counts["PASS"] > 0 and counts["BLOCK"] == 0:
        status = "🟢 Có mô hình đạt điều kiện"
    else:
        status = "🟡 Đang tích lũy bằng chứng"
    return {
        "status": status, "total": len(reports), "pass": counts["PASS"],
        "caution": counts["CAUTION"], "block": counts["BLOCK"],
        "avg_edge_pp": round(float(np.mean(edges)), 2) if edges else None,
        "rules": {"min_holdout_samples": MIN_HOLDOUT_SAMPLES, "min_paper_t5_samples": MIN_PAPER_SAMPLES},
    }
