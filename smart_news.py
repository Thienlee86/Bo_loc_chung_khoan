"""Làm sạch, gắn thực thể và chấm tác động tin tức tài chính Việt Nam."""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, timezone

import pandas as pd

from sector_analysis import TICKER_SECTORS, sector_for_ticker


TICKER_ALIASES = {
    "ACB": ["acb", "á châu"], "BID": ["bidv", "bid"],
    "BSR": ["lọc hóa dầu bình sơn", "bình sơn", "bsr"],
    "BVH": ["bảo việt", "bvh"], "CTG": ["vietinbank", "ctg"],
    "FPT": ["tập đoàn fpt", "fpt corp", "fpt"], "GAS": ["pv gas", "gas"],
    "GVR": ["cao su việt nam", "gvr"], "HDB": ["hdbank", "hdb"],
    "HPG": ["hòa phát", "hoà phát", "hpg"], "MBB": ["mb bank", "mbb"],
    "MSN": ["masan", "msn"], "MWG": ["thế giới di động", "mwg"],
    "PLX": ["petrolimex", "plx"], "POW": ["pv power", "pow"],
    "SAB": ["sabeco", "sab"], "SSB": ["seabank", "ssb"],
    "SSI": ["chứng khoán ssi", "ssi"], "STB": ["sacombank", "stb"],
    "TCB": ["techcombank", "tcb"], "TPB": ["tpbank", "tpb"],
    "VCB": ["vietcombank", "vcb"], "VHM": ["vinhomes", "vhm"],
    "VIB": ["ngân hàng vib", "vib"], "VIC": ["vingroup", "vic"],
    "VJC": ["vietjet", "vjc"], "VNM": ["vinamilk", "vnm"],
    "VPB": ["vpbank", "vpb"], "VPL": ["vinpearl", "vpl"],
    "VRE": ["vincom retail", "vre"],
}

EVENT_KEYWORDS = {
    "Kết quả kinh doanh": ["doanh thu", "lợi nhuận", "báo lãi", "báo lỗ", "kết quả kinh doanh"],
    "Cổ tức & phát hành": ["cổ tức", "phát hành", "chia thưởng", "quyền mua"],
    "Pháp lý & quản trị": ["thanh tra", "điều tra", "khởi tố", "xử phạt", "từ nhiệm", "bổ nhiệm"],
    "Dự án & hợp đồng": ["trúng thầu", "hợp đồng", "dự án", "khởi công", "bàn giao"],
    "Giao dịch cổ đông": ["mua vào", "bán ra", "cổ đông lớn", "mua cổ phiếu", "bán cổ phiếu"],
    "Vĩ mô & chính sách": ["lãi suất", "tỷ giá", "ngân hàng nhà nước", "nghị định", "thuế", "chính sách"],
    "Giá hàng hóa": ["giá dầu", "giá thép", "giá vàng", "giá cao su", "giá khí"],
}

EVENT_WEIGHTS = {
    "Kết quả kinh doanh": 1.25, "Pháp lý & quản trị": 1.25,
    "Dự án & hợp đồng": 1.10, "Vĩ mô & chính sách": 1.10,
    "Giá hàng hóa": 1.05, "Cổ tức & phát hành": 1.00,
    "Giao dịch cổ đông": 0.90, "Tin chung": 0.75,
}

POSITIVE_PHRASES = [
    "tăng trưởng", "vượt kế hoạch", "lợi nhuận kỷ lục", "lãi lớn", "bứt phá",
    "hưởng lợi", "mua ròng", "trúng thầu", "nâng hạng", "phục hồi", "cổ tức cao",
]
NEGATIVE_PHRASES = [
    "thua lỗ", "nợ xấu", "bị phạt", "giảm mạnh", "bán tháo", "sụt giảm",
    "bán ròng", "lao dốc", "sai phạm", "thanh tra", "điều tra", "đình chỉ",
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "").lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def deduplicate_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """Loại bài sao chép dựa trên fingerprint tiêu đề không dấu."""
    if news_df is None or news_df.empty:
        return pd.DataFrame(columns=getattr(news_df, "columns", []))
    result = news_df.copy()
    result["_fingerprint"] = result["title"].map(normalize_text).map(
        lambda value: " ".join(sorted(set(value.split())))
    )
    result = result[result["_fingerprint"].str.len() > 0]
    result = result.drop_duplicates("_fingerprint", keep="first")
    return result.drop(columns="_fingerprint").reset_index(drop=True)


def mentioned_tickers(text: str) -> list[str]:
    normalized = f" {normalize_text(text)} "
    found = []
    for ticker, aliases in TICKER_ALIASES.items():
        if any(f" {normalize_text(alias)} " in normalized for alias in aliases):
            found.append(ticker)
    return found


def classify_event(text: str) -> str:
    normalized = normalize_text(text)
    best_event, best_hits = "Tin chung", 0
    for event, keywords in EVENT_KEYWORDS.items():
        hits = sum(normalize_text(keyword) in normalized for keyword in keywords)
        if hits > best_hits:
            best_event, best_hits = event, hits
    return best_event


def score_impact(text: str) -> dict:
    normalized = normalize_text(text)
    positive = [p for p in POSITIVE_PHRASES if normalize_text(p) in normalized]
    negative = [p for p in NEGATIVE_PHRASES if normalize_text(p) in normalized]
    raw = max(-2, min(2, len(positive) - len(negative)))
    label = "Tích cực" if raw > 0 else "Tiêu cực" if raw < 0 else "Trung lập"
    level = "Mạnh" if abs(raw) >= 2 else "Vừa" if abs(raw) == 1 else "Thấp/chưa rõ"
    return {"score": raw, "label": label, "impact_level": level, "matched_words": positive + negative}


def _recency_weight(value, now=None) -> float:
    now = now or datetime.now(timezone.utc)
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return 0.5
    age_days = max(0.0, (now - parsed.to_pydatetime()).total_seconds() / 86400)
    return max(0.2, math.exp(-age_days / 3.0))


def enrich_news(news_df: pd.DataFrame, now=None) -> pd.DataFrame:
    result = deduplicate_news(news_df)
    if result.empty:
        return result
    texts = (result["title"].fillna("") + " " + result["summary"].fillna(""))
    result["tickers"] = texts.map(mentioned_tickers)
    result["sectors"] = result["tickers"].map(lambda items: sorted({sector_for_ticker(t) for t in items}))
    result["event_type"] = texts.map(classify_event)
    impacts = texts.map(score_impact)
    result["sentiment_score"] = impacts.map(lambda item: item["score"])
    result["sentiment_label"] = impacts.map(lambda item: item["label"])
    result["impact_level"] = impacts.map(lambda item: item["impact_level"])
    result["matched_words"] = impacts.map(lambda item: item["matched_words"])
    result["recency_weight"] = result.get("published", pd.Series("", index=result.index)).map(
        lambda value: _recency_weight(value, now)
    )
    result["event_weight"] = result["event_type"].map(EVENT_WEIGHTS).fillna(0.75)
    return result.reset_index(drop=True)


def build_sector_news_scores(news_df: pd.DataFrame, now=None) -> dict[str, dict]:
    enriched = enrich_news(news_df, now=now)
    buckets = {sector: [] for sector in set(TICKER_SECTORS.values())}
    for row in enriched.to_dict("records"):
        weight = float(row["recency_weight"] * row["event_weight"])
        for sector in row["sectors"]:
            buckets.setdefault(sector, []).append((float(row["sentiment_score"]), weight, row["event_type"]))

    output = {}
    for sector, values in buckets.items():
        if not values:
            continue
        denominator = sum(weight for _, weight, _ in values)
        net = sum(score * weight for score, weight, _ in values) / denominator if denominator else 0.0
        output[sector] = {
            "score": round(max(0.0, min(100.0, 50 + 25 * net)), 1),
            "article_count": len(values), "net_impact": round(net, 2),
            "events": sorted({event for _, _, event in values}),
        }
    return output
