"""
Module lấy tin tức tài chính VN qua RSS và chấm điểm sentiment bằng lexicon
(từ điển từ khoá tích cực/tiêu cực). Hoàn toàn miễn phí, chạy offline sau khi
tải RSS - không gọi API trả phí nào.

LƯU Ý: đây là phương án THAM KHẢO, không phải phân tích ngữ nghĩa thực sự.
Sentiment tính bằng cách đếm từ khoá xuất hiện, không hiểu ngữ cảnh/phủ định.
"""

import re
import feedparser
import pandas as pd
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Nguồn RSS (đã kiểm tra URL thật, có thể thêm/bớt nguồn tại đây)
# ---------------------------------------------------------------------------

RSS_SOURCES = {
    "CafeF - Chứng khoán": "https://cafef.vn/thi-truong-chung-khoan.rss",
    "CafeF - Doanh nghiệp": "https://cafef.vn/doanh-nghiep.rss",
    "CafeBiz - Chứng khoán": "https://cafebiz.vn/rss/chung-khoan.rss",
}

# ---------------------------------------------------------------------------
# Ánh xạ mã CK -> tên công ty & từ khoá liên quan (mở rộng dần theo nhu cầu)
# ---------------------------------------------------------------------------

TICKER_ALIASES = {
    "VNM": ["vinamilk", "vnm"],
    "VCB": ["vietcombank", "vcb"],
    "HPG": ["hoà phát", "hoa phat", "hpg", "tập đoàn hòa phát"],
    "FPT": ["fpt", "tập đoàn fpt", "fpt corp"],
    "VIC": ["vingroup", "vic"],
    "VHM": ["vinhomes", "vhm"],
    "MSN": ["masan", "msn"],
    "MWG": ["thế giới di động", "mwg", "the gioi di dong"],
    "TCB": ["techcombank", "tcb"],
    "MBB": ["mb bank", "ngân hàng quân đội", "mbb"],
    "GAS": ["pv gas", "gas"],
    "SSI": ["chứng khoán ssi", "ssi"],
    "STB": ["sacombank", "stb"],
    "CTG": ["vietinbank", "ctg"],
    "BID": ["bidv", "bid"],
}

# ---------------------------------------------------------------------------
# Lexicon sentiment tài chính tiếng Việt (mở rộng dần)
# ---------------------------------------------------------------------------

POSITIVE_WORDS = [
    "tăng trưởng", "vượt kế hoạch", "lợi nhuận kỷ lục", "lãi lớn", "lãi đậm",
    "khởi sắc", "bứt phá", "tích cực", "hưởng lợi", "kỳ vọng tăng", "mua ròng",
    "vượt mốc", "cao nhất", "kỷ lục", "tăng mạnh", "khả quan", "thuận lợi",
    "cổ tức cao", "mở rộng", "ký kết hợp đồng lớn", "trúng thầu", "nâng hạng",
    "dòng tiền mạnh", "thanh khoản tốt", "vượt đỉnh", "phục hồi", "tăng vọt",
]

NEGATIVE_WORDS = [
    "thua lỗ", "nợ xấu", "bị phạt", "giảm mạnh", "bán tháo", "rủi ro", "sụt giảm",
    "thấp hơn kỳ vọng", "cắt giảm", "khó khăn", "bán ròng", "giảm sâu", "lao dốc",
    "cảnh báo", "kiện tụng", "sai phạm", "thanh tra", "điều tra", "phá sản",
    "nợ vay tăng", "áp lực", "tiêu cực", "mất thanh khoản", "xuống đáy", "giảm sốc",
    "hủy niêm yết", "đình chỉ",
]


# ---------------------------------------------------------------------------
# SENTIMENT BẰNG CLAUDE API (nâng cấp — trả phí, chính xác hơn lexicon)
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # model rẻ nhất, đủ tốt cho phân loại sentiment

CLAUDE_SENTIMENT_PROMPT = """Bạn là chuyên gia phân tích tin tức tài chính Việt Nam.
Đọc bài tin dưới đây và đánh giá mức độ ảnh hưởng đến giá cổ phiếu mã {ticker}.

Tiêu đề: {title}
Tóm tắt: {summary}

Trả lời CHỈ theo đúng định dạng JSON sau, không thêm chữ nào khác:
{{"label": "Tích cực" | "Tiêu cực" | "Trung lập", "score": số từ -2 đến 2, "reason": "giải thích ngắn gọn 1 câu bằng tiếng Việt"}}
"""


def score_sentiment_claude(title: str, summary: str, ticker: str, client) -> dict:
    """Chấm sentiment bằng Claude API. `client` là anthropic.Anthropic() đã khởi tạo.
    Trả về dict cùng format với score_sentiment() để 2 hàm thay thế được cho nhau."""
    import json

    prompt = CLAUDE_SENTIMENT_PROMPT.format(ticker=ticker, title=title, summary=summary or "(không có tóm tắt)")

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Claude đôi khi bọc JSON trong ```json ... ``` dù đã dặn không làm vậy
    raw = raw.replace("```json", "").replace("```", "").strip()

    parsed = json.loads(raw)
    return {
        "label": parsed["label"],
        "score": parsed["score"],
        "pos_hits": [], "neg_hits": [],  # giữ key cho tương thích, Claude không trả từ khoá
        "reason": parsed.get("reason", ""),
    }


def fetch_all_news(max_per_source: int = 25) -> pd.DataFrame:
    """Lấy tin từ toàn bộ nguồn RSS, trả về DataFrame gộp."""
    rows = []
    for source_name, url in RSS_SOURCES.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_source]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                link = getattr(entry, "link", "")
                published = getattr(entry, "published", "")
                rows.append({
                    "source": source_name, "title": title, "summary": summary,
                    "link": link, "published": published,
                })
        except Exception:
            # Một nguồn lỗi không nên làm sập toàn bộ app
            continue
    return pd.DataFrame(rows)


def score_sentiment(text: str) -> dict:
    """Chấm điểm sentiment bằng đếm từ khoá. Trả về nhãn + điểm + từ khoá khớp."""
    text_lower = text.lower()
    pos_hits = [w for w in POSITIVE_WORDS if w in text_lower]
    neg_hits = [w for w in NEGATIVE_WORDS if w in text_lower]

    score = len(pos_hits) - len(neg_hits)
    if score > 0:
        label = "Tích cực"
    elif score < 0:
        label = "Tiêu cực"
    else:
        label = "Trung lập"

    return {"label": label, "score": score, "pos_hits": pos_hits, "neg_hits": neg_hits}


def match_ticker(text: str, ticker: str) -> bool:
    """Kiểm tra bài tin có nhắc đến mã CK này không (theo alias)."""
    text_lower = text.lower()
    aliases = TICKER_ALIASES.get(ticker.upper(), [ticker.lower()])
    return any(alias in text_lower for alias in aliases)


def get_news_for_ticker(ticker: str, news_df: pd.DataFrame, use_claude: bool = False, claude_client=None) -> pd.DataFrame:
    """Lọc tin liên quan 1 mã CK và gắn điểm sentiment cho từng tin.

    use_claude=False (mặc định): dùng lexicon miễn phí, offline.
    use_claude=True: dùng Claude API — chính xác hơn nhưng tốn phí nhỏ, cần
    truyền claude_client (anthropic.Anthropic đã khởi tạo với API key)."""
    if news_df.empty:
        return news_df

    mask = news_df.apply(
        lambda r: match_ticker(f"{r['title']} {r['summary']}", ticker), axis=1
    )
    filtered = news_df[mask].copy()

    if filtered.empty:
        return filtered

    if use_claude and claude_client is not None:
        results = []
        for _, r in filtered.iterrows():
            try:
                results.append(score_sentiment_claude(r["title"], r["summary"], ticker, claude_client))
            except Exception:
                # Nếu API lỗi (hết quota, mất mạng...), fallback về lexicon cho tin đó
                results.append(score_sentiment(f"{r['title']} {r['summary']}"))
    else:
        results = [score_sentiment(f"{r['title']} {r['summary']}") for _, r in filtered.iterrows()]

    filtered["sentiment_label"] = [r["label"] for r in results]
    filtered["sentiment_score"] = [r["score"] for r in results]
    filtered["matched_words"] = [r.get("pos_hits", []) + r.get("neg_hits", []) for r in results]
    filtered["reason"] = [r.get("reason", "") for r in results]
    return filtered.reset_index(drop=True)


def summarize_sentiment(filtered_news: pd.DataFrame) -> dict:
    """Tổng hợp điểm sentiment tổng quan từ danh sách tin đã lọc."""
    if filtered_news.empty:
        return {"n_news": 0, "n_pos": 0, "n_neg": 0, "n_neutral": 0, "overall": "Chưa có tin"}

    n_pos = int((filtered_news["sentiment_label"] == "Tích cực").sum())
    n_neg = int((filtered_news["sentiment_label"] == "Tiêu cực").sum())
    n_neutral = int((filtered_news["sentiment_label"] == "Trung lập").sum())

    if n_pos > n_neg:
        overall = "Nghiêng tích cực"
    elif n_neg > n_pos:
        overall = "Nghiêng tiêu cực"
    else:
        overall = "Cân bằng / trung lập"

    return {
        "n_news": len(filtered_news), "n_pos": n_pos, "n_neg": n_neg,
        "n_neutral": n_neutral, "overall": overall,
    }
