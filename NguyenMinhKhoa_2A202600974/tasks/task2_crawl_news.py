"""
Task 2 - Crawl real news articles into data/landing/news.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "landing" / "news"


ARTICLE_URLS = [
    "https://vietnamnet.vn/tu-vu-huu-tin-bi-bat-vi-dung-ma-tuy-nghe-si-can-trong-khi-phat-ngon-2030279.html",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://tienphong.vn/vu-cong-tri-bi-bat-duoi-goc-nhin-chuyen-gia-dieu-tra-ma-tuy-post1763699.tpo",
    "https://tienphong.vn/rapper-mr-nhan-vua-bi-bat-trong-duong-day-ma-tuy-140-nguoi-la-ai-post1847153.tpo",
    "https://vietnamnet.vn/van-hoa-giai-tri/toan-canh-vu-bat-tam-giam-long-nhat-va-son-ngoc-sk0008VN.html",
]


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


async def crawl_article(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        metadata = getattr(result, "metadata", {}) or {}
        title = metadata.get("title") or getattr(result, "title", None) or "Unknown"
        content = (
            getattr(result, "markdown", "")
            or getattr(result, "fit_markdown", "")
            or getattr(result, "cleaned_html", "")
        )
        return {
            "url": url,
            "title": title,
            "date_crawled": datetime.now().isoformat(),
            "content_markdown": content,
        }


async def crawl_all():
    setup_directory()
    for i, url in enumerate(ARTICLE_URLS, start=1):
        article = await crawl_article(url)
        out = DATA_DIR / f"article_{i:02d}.json"
        out.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(crawl_all())
