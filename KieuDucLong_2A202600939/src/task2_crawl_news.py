"""Task 2 - Crawl or load news articles for the individual RAG pipeline.

Network crawling is optional in the lab environment.  The repository already
contains curated article_*.json files in data/landing/news, so this module first
loads those local artifacts.  If a new URL is requested and crawl4ai is
installed, it can crawl the URL; otherwise it returns a clear fallback record
instead of raising NotImplementedError.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# URLs are populated from the existing curated JSON corpus at runtime.  Keeping
# this list import-safe avoids network work during automated tests.
ARTICLE_URLS: list[str] = []


def setup_directory() -> Path:
    """Create data/landing/news and return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_local_articles() -> list[dict[str, Any]]:
    """Load curated local article JSON files."""
    setup_directory()
    articles: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("article_*.json")):
        try:
            item = _read_json(path)
            item.setdefault("id", path.stem)
            item.setdefault("crawl_date", datetime.now(timezone.utc).date().isoformat())
            item["_path"] = str(path)
            articles.append(item)
        except json.JSONDecodeError:
            continue
    return articles


def get_article_urls() -> list[str]:
    """Return source URLs from local data, falling back to ARTICLE_URLS."""
    urls = [str(item.get("url")) for item in load_local_articles() if item.get("url")]
    return urls or ARTICLE_URLS


async def crawl_article(url: str) -> dict[str, Any]:
    """Return article metadata/content for a URL.

    The function prefers matching local JSON data for deterministic tests and
    demos.  It attempts crawl4ai only for URLs that are not already present.
    """
    for item in load_local_articles():
        if item.get("url") == url:
            return {k: v for k, v in item.items() if k != "_path"}

    try:
        from crawl4ai import AsyncWebCrawler  # type: ignore
    except Exception:
        return {
            "url": url,
            "title": "Uncrawled article",
            "publisher": "unknown",
            "published_date": "",
            "crawl_date": datetime.now(timezone.utc).date().isoformat(),
            "topic": "ma tuy",
            "summary": "crawl4ai is not installed; local curated articles are used for the lab.",
            "content": "crawl4ai is not installed; local curated articles are used for the lab.",
        }

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        metadata = getattr(result, "metadata", {}) or {}
        markdown = getattr(result, "markdown", "") or ""
        return {
            "url": url,
            "title": metadata.get("title", "Unknown title"),
            "publisher": metadata.get("site_name", "unknown"),
            "published_date": metadata.get("published_time", ""),
            "crawl_date": datetime.now(timezone.utc).date().isoformat(),
            "topic": "ma tuy",
            "summary": markdown[:500],
            "content": markdown,
        }


async def crawl_all(urls: list[str] | None = None) -> list[Path]:
    """Crawl/load all URLs and save them as article_XX.json files."""
    setup_directory()
    target_urls = urls or get_article_urls()
    saved: list[Path] = []
    for i, url in enumerate(target_urls, 1):
        article = await crawl_article(url)
        filepath = DATA_DIR / f"article_{i:03d}.json"
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append(filepath)
    return saved


if __name__ == "__main__":
    urls = get_article_urls()
    if not urls:
        print("No URLs found. Add article_*.json files or populate ARTICLE_URLS.")
    else:
        saved_paths = asyncio.run(crawl_all(urls))
        print(f"Saved {len(saved_paths)} article files to {DATA_DIR}")
