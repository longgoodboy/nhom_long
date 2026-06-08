"""
Task 2 — Crawl bài báo thật về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Lưu output vào data/landing/news/.
    3. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled,
       content_markdown).

Cách hoạt động:
    - Dùng `requests` để fetch HTML.
    - Dùng `BeautifulSoup` để bóc title + body theo selector chuyên biệt
      cho từng tờ báo (vnexpress / tuoitre / thanhnien / dantri / vietnamnet).
    - Fallback xuống `og:description` + `<article> > p` nếu selector chính fail.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

ARTICLE_URLS = [
    "https://baomoi.com/loat-sao-viet-tung-nga-ngua-vi-ma-tuy-cu-truot-dai-khong-loi-thoat-c52826789.epi",
    "https://tienphong.vn/lien-tiep-nghe-si-dung-chat-cam-post1842599.tpo",
    "https://nld.com.vn/showbiz-viet-nhung-nghe-si-gay-soc-vi-be-boi-ma-tuy-196250725113547841.htm",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
    "https://baomoi.com/sao-viet-tieu-tan-su-nghiep-vi-lien-quan-den-ma-tuy-c55216942.epi",
]


@dataclass
class SiteSelectors:
    title: tuple[str, ...]
    body: tuple[str, ...]


SITE_SELECTORS: dict[str, SiteSelectors] = {
    "tienphong.vn": SiteSelectors(
        title=("h1.article__title", "h1.title-detail", "h1.detail-title"),
        body=(".article__body.zce-content-body.cms-body", ".article__body", ".cms-body"),
    ),
    "nld.com.vn": SiteSelectors(
        title=("h1.title-detail", "h1.article-title", "h1.detail__title"),
        body=(".detail__cmain-main", ".detail__main", ".detail__content", ".detail-content"),
    ),
    "vov.vn": SiteSelectors(
        title=("h1.title-detail", "h1.article-title", "h1[class*='title']"),
        body=(".article-content", "article.detail", ".content-detail", ".vovvn-body"),
    ),
    # baomoi.com is an aggregator that 301-redirects to the original publisher.
    # requests follows redirects automatically — selectors below cover the
    # publishers we saw redirected to:
    "gocnhinphaply.nguoiduatin.vn": SiteSelectors(
        title=("h1.article-title", "h1.title-detail", "h1[class*='title']"),
        body=(".article-content", ".dt-news__body", ".article-detail"),
    ),
    "vtcnews.vn": SiteSelectors(
        title=("h1.detail-title", "h1.title-detail", "h1[class*='title']"),
        body=(".edittor-content", ".content-wrapper", "article.nd-detail"),
    ),
}


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def select_first(soup: BeautifulSoup, selectors: tuple[str, ...]) -> str | None:
    for sel in selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text("\n", strip=True)
    return None


def site_of(url: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", url).split("/", 1)[0]


def extract_article(html: str, url: str) -> tuple[str, str]:
    """Trả về (title, body_markdown). Raise nếu không bóc được."""
    soup = BeautifulSoup(html, "html.parser")
    selectors = SITE_SELECTORS.get(site_of(url))

    title = None
    body = None
    if selectors:
        title = select_first(soup, selectors.title)
        body = select_first(soup, selectors.body)

    if not title:
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title = og_title["content"]
        elif soup.title and soup.title.string:
            title = soup.title.string.strip()

    if not body or len(body) < 200:
        article = soup.find("article")
        if article:
            paras = [p.get_text(" ", strip=True) for p in article.find_all("p")]
            joined = "\n\n".join(p for p in paras if p)
            if len(joined) > 200:
                body = joined

    if not body or len(body) < 200:
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            body = og_desc["content"]

    if not title or not body:
        raise RuntimeError(f"Không bóc được title/body từ {url}")

    body_md = re.sub(r"\n{3,}", "\n\n", body.strip())
    return title.strip(), body_md


_BAOMOI_ORIGINAL_URL_RE = re.compile(r'"originalUrl":"(https://[^"]+)"')


def _fetch(url: str) -> requests.Response:
    resp = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "vi,en;q=0.9",
        },
        timeout=30,
        allow_redirects=True,
    )
    resp.raise_for_status()
    return resp


def crawl_article(url: str) -> dict:
    """Crawl 1 bài báo. Tự follow redirect baomoi.com → publisher gốc."""
    resp = _fetch(url)
    final_url = resp.url
    html = resp.text

    if "baomoi.com" in site_of(final_url):
        match = _BAOMOI_ORIGINAL_URL_RE.search(html)
        if not match:
            raise RuntimeError(f"baomoi page không có originalUrl: {final_url}")
        publisher_url = match.group(1)
        print(f"  ↻ baomoi → {publisher_url}")
        resp = _fetch(publisher_url)
        final_url = resp.url
        html = resp.text

    title, body_md = extract_article(html, final_url)
    return {
        "url": url,
        "final_url": final_url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": f"# {title}\n\n{body_md}\n",
    }


def crawl_all() -> None:
    setup_directory()
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] {url}")
        try:
            article = crawl_article(url)
        except Exception as exc:
            print(f"  ✗ Lỗi: {exc}")
            continue
        filepath = DATA_DIR / f"article_{i:02d}.json"
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  ✓ {filepath.name} — {article['title'][:80]} ({len(article['content_markdown']):,} chars)")


if __name__ == "__main__":
    crawl_all()
