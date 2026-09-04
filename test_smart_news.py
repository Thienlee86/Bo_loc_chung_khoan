from datetime import datetime, timezone

import pandas as pd

from smart_news import build_sector_news_scores, classify_event, deduplicate_news, enrich_news, mentioned_tickers


def _news(rows):
    return pd.DataFrame(rows, columns=["source", "title", "summary", "link", "published"])


def test_duplicate_headlines_from_different_sources_count_once():
    data = _news([
        ["A", "FPT báo lãi tăng trưởng mạnh!", "", "a", "2026-09-04"],
        ["B", "FPT báo lãi: tăng trưởng mạnh", "", "b", "2026-09-04"],
    ])
    assert len(deduplicate_news(data)) == 1


def test_entities_and_event_are_detected():
    text = "Vietcombank công bố lợi nhuận tăng trưởng và cổ tức"
    assert "VCB" in mentioned_tickers(text)
    assert classify_event(text) == "Kết quả kinh doanh"


def test_enrichment_adds_explainable_fields():
    data = _news([["A", "Hòa Phát trúng thầu hợp đồng lớn", "", "a", "2026-09-04"]])
    result = enrich_news(data, now=datetime(2026, 9, 4, tzinfo=timezone.utc)).iloc[0]
    assert result["tickers"] == ["HPG"]
    assert result["sectors"] == ["Thép"]
    assert result["event_type"] == "Dự án & hợp đồng"
    assert result["sentiment_score"] > 0


def test_sector_news_score_is_bounded_and_has_evidence_count():
    data = _news([["A", "FPT lợi nhuận kỷ lục, tăng trưởng", "", "a", "2026-09-04"]])
    result = build_sector_news_scores(data, now=datetime(2026, 9, 4, tzinfo=timezone.utc))["Công nghệ"]
    assert 50 < result["score"] <= 100
    assert result["article_count"] == 1
