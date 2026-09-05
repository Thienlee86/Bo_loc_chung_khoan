"""Xếp hạng sức mạnh nhóm ngành từ dữ liệu giá/khối lượng của từng mã."""

from __future__ import annotations

import numpy as np
import pandas as pd


# Phân nhóm phục vụ phân tích dòng tiền, ưu tiên cách gọi dễ hiểu trên app.
# Mã chưa có trong bảng vẫn được giữ lại dưới nhóm "Chưa phân loại".
TICKER_SECTORS = {
    "ACB": "Ngân hàng", "BID": "Ngân hàng", "CTG": "Ngân hàng", "EIB": "Ngân hàng",
    "HDB": "Ngân hàng", "LPB": "Ngân hàng", "MBB": "Ngân hàng", "MSB": "Ngân hàng",
    "OCB": "Ngân hàng", "SHB": "Ngân hàng", "SSB": "Ngân hàng", "STB": "Ngân hàng",
    "TCB": "Ngân hàng", "TPB": "Ngân hàng", "VCB": "Ngân hàng", "VIB": "Ngân hàng", "VPB": "Ngân hàng",
    "SSI": "Chứng khoán", "HCM": "Chứng khoán", "MBS": "Chứng khoán", "ORS": "Chứng khoán",
    "SHS": "Chứng khoán", "VCI": "Chứng khoán", "VDS": "Chứng khoán", "VIX": "Chứng khoán", "VND": "Chứng khoán",
    "BSR": "Dầu khí", "GAS": "Dầu khí", "PLX": "Dầu khí", "PVD": "Dầu khí", "PVS": "Dầu khí", "PVT": "Dầu khí", "PET": "Dầu khí",
    "VHM": "Bất động sản", "VIC": "Bất động sản", "VRE": "Bất động sản", "DXG": "Bất động sản",
    "DIG": "Bất động sản", "NLG": "Bất động sản", "KDH": "Bất động sản", "PDR": "Bất động sản", "TCH": "Bất động sản", "VPI": "Bất động sản", "AGG": "Bất động sản",
    "BCM": "Khu công nghiệp", "IDC": "Khu công nghiệp", "KBC": "Khu công nghiệp", "GVR": "Cao su & KCN", "PHR": "Cao su & KCN", "VGC": "Vật liệu & KCN",
    "HPG": "Thép", "HSG": "Thép", "NKG": "Thép", "KSB": "Vật liệu xây dựng", "BMP": "Vật liệu xây dựng", "AAA": "Nhựa & bao bì",
    "FPT": "Công nghệ", "CMG": "Công nghệ", "DGW": "Phân phối công nghệ", "FRT": "Bán lẻ", "MWG": "Bán lẻ",
    "MSN": "Tiêu dùng", "SAB": "Tiêu dùng", "VNM": "Tiêu dùng", "SBT": "Thực phẩm",
    "PAN": "Nông nghiệp", "DBC": "Nông nghiệp", "BAF": "Nông nghiệp", "HAG": "Nông nghiệp", "ANV": "Thủy sản", "VHC": "Thủy sản",
    "DCM": "Phân bón", "DPM": "Phân bón", "DGC": "Hóa chất",
    "POW": "Điện", "NT2": "Điện", "PC1": "Điện", "REE": "Điện & hạ tầng", "HDG": "Điện & bất động sản",
    "GEX": "Thiết bị điện", "GMD": "Cảng & logistics", "HAH": "Vận tải biển", "VOS": "Vận tải biển", "VSC": "Cảng & logistics",
    "CII": "Hạ tầng", "HHV": "Hạ tầng", "VCG": "Xây dựng & hạ tầng",
    "BVH": "Bảo hiểm", "VJC": "Hàng không", "VPL": "Du lịch & giải trí",
}


def sector_for_ticker(ticker: str) -> str:
    return TICKER_SECTORS.get(ticker.upper(), "Chưa phân loại")


def _safe_return(close: pd.Series, periods: int) -> float:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if len(clean) <= periods or clean.iloc[-periods - 1] == 0:
        return np.nan
    return float(clean.iloc[-1] / clean.iloc[-periods - 1] - 1)


def _ticker_snapshot(ticker: str, raw: pd.DataFrame, benchmark_ret20: float) -> dict | None:
    if raw is None or raw.empty or len(raw) < 55:
        return None
    close = pd.to_numeric(raw["close"], errors="coerce")
    volume = pd.to_numeric(raw["volume"], errors="coerce")
    ma20, ma50 = close.rolling(20).mean(), close.rolling(50).mean()
    vol20 = volume.rolling(20).mean()
    latest_volume_ratio = float(volume.iloc[-1] / vol20.iloc[-1]) if vol20.iloc[-1] > 0 else np.nan
    ret20 = _safe_return(close, 20)
    return {
        "ticker": ticker,
        "sector": sector_for_ticker(ticker),
        "ret_5": _safe_return(close, 5),
        "ret_20": ret20,
        "above_ma20": bool(close.iloc[-1] > ma20.iloc[-1]),
        "above_ma50": bool(close.iloc[-1] > ma50.iloc[-1]),
        "volume_ratio": latest_volume_ratio,
        "is_leader": bool(np.isfinite(ret20) and ret20 > benchmark_ret20 and close.iloc[-1] > ma20.iloc[-1]),
    }


def _percentile_score(values: pd.Series) -> pd.Series:
    """Đổi chỉ tiêu thành điểm 0-100; nhóm bằng nhau nhận cùng điểm giữa."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() <= 1 or numeric.nunique(dropna=True) <= 1:
        return pd.Series(50.0, index=values.index)
    return numeric.rank(pct=True, method="average").fillna(0.5) * 100


def classify_sector(score: float) -> str:
    if score >= 75:
        return "Dẫn dắt mạnh"
    if score >= 60:
        return "Đang cải thiện"
    if score >= 45:
        return "Trung tính / tích lũy"
    if score >= 30:
        return "Suy yếu"
    return "Điều chỉnh mạnh"


def calculate_sector_rankings(
    histories: dict[str, pd.DataFrame],
    benchmark: pd.DataFrame | None,
    sector_news_scores: dict[str, dict] | None = None,
) -> list[dict]:
    """Tính Sector Score 0-100 từ năm thành phần độc lập.

    Trọng số giai đoạn 2 (chưa có tin tức): sức mạnh tương đối 30%, độ rộng
    25%, dòng tiền 20%, động lượng 15%, cổ phiếu dẫn dắt 10%.
    """
    benchmark_ret20 = _safe_return(benchmark["close"], 20) if benchmark is not None and not benchmark.empty else 0.0
    if not np.isfinite(benchmark_ret20):
        benchmark_ret20 = 0.0

    snapshots = []
    for ticker, raw in histories.items():
        snapshot = _ticker_snapshot(ticker, raw, benchmark_ret20)
        if snapshot:
            snapshots.append(snapshot)
    if not snapshots:
        return []

    stocks = pd.DataFrame(snapshots)
    grouped = stocks.groupby("sector", dropna=False).agg(
        stock_count=("ticker", "count"),
        return_5d=("ret_5", "mean"),
        return_20d=("ret_20", "mean"),
        breadth_ma20=("above_ma20", "mean"),
        breadth_ma50=("above_ma50", "mean"),
        volume_ratio=("volume_ratio", "mean"),
        leader_ratio=("is_leader", "mean"),
    ).reset_index()

    grouped["relative_strength_20d"] = grouped["return_20d"] - benchmark_ret20
    grouped["relative_strength_score"] = _percentile_score(grouped["relative_strength_20d"])
    grouped["breadth_score"] = (grouped["breadth_ma20"] + grouped["breadth_ma50"]) * 50
    grouped["flow_score"] = ((grouped["volume_ratio"] - 0.7) / 0.8 * 100).clip(0, 100).fillna(50)
    momentum_raw = grouped["return_5d"].fillna(0) * 0.4 + grouped["return_20d"].fillna(0) * 0.6
    grouped["momentum_score"] = _percentile_score(momentum_raw)
    grouped["leadership_score"] = grouped["leader_ratio"] * 100
    grouped["technical_score"] = (
        grouped["relative_strength_score"] * 0.30
        + grouped["breadth_score"] * 0.25
        + grouped["flow_score"] * 0.20
        + grouped["momentum_score"] * 0.15
        + grouped["leadership_score"] * 0.10
    ).clip(0, 100)
    sector_news_scores = sector_news_scores or {}
    grouped["news_score"] = grouped["sector"].map(
        lambda sector: sector_news_scores.get(sector, {}).get("score", np.nan)
    )
    grouped["news_count"] = grouped["sector"].map(
        lambda sector: int(sector_news_scores.get(sector, {}).get("article_count", 0))
    )
    # Khi có tin: kỹ thuật 90% + tin tức 10%. Khi không có tin, không tự gán trung lập.
    grouped["score"] = np.where(
        grouped["news_count"] > 0,
        grouped["technical_score"] * 0.90 + grouped["news_score"] * 0.10,
        grouped["technical_score"],
    )
    grouped["score"] = grouped["score"].clip(0, 100).round(1)
    grouped["status"] = grouped["score"].map(classify_sector)
    grouped = grouped.sort_values(["score", "relative_strength_20d"], ascending=False)

    result = []
    for row in grouped.to_dict("records"):
        result.append({
            "sector": row["sector"], "score": float(row["score"]), "status": row["status"],
            "stock_count": int(row["stock_count"]), "return_5d_pct": round(float(row["return_5d"] * 100), 2),
            "return_20d_pct": round(float(row["return_20d"] * 100), 2),
            "relative_strength_20d_pct": round(float(row["relative_strength_20d"] * 100), 2),
            "breadth_ma20_pct": round(float(row["breadth_ma20"] * 100), 1),
            "breadth_ma50_pct": round(float(row["breadth_ma50"] * 100), 1),
            "volume_ratio": round(float(row["volume_ratio"]), 2),
            "leader_ratio_pct": round(float(row["leader_ratio"] * 100), 1),
            "news_score": round(float(row["news_score"]), 1) if np.isfinite(row["news_score"]) else None,
            "news_count": int(row["news_count"]),
        })
    return result


def attach_score_changes(current: list[dict], previous: list[dict] | None) -> list[dict]:
    """So sánh với lần quét trước nếu có; không tự suy diễn khi thiếu lịch sử."""
    previous_scores = {r.get("sector"): r.get("score") for r in (previous or [])}
    output = []
    for row in current:
        item = dict(row)
        old_score = previous_scores.get(row["sector"])
        item["score_change"] = round(row["score"] - old_score, 1) if isinstance(old_score, (int, float)) else None
        output.append(item)
    return output
