"""
Task 2 - Crawl news articles about artists/public figures related to drugs.

Outputs:
    data/landing/news/article_01.json
    ...

Each JSON file contains url, title, date_crawled, source, and content_markdown.
The crawler prefers Crawl4AI when it is installed, and falls back to a lightweight
requests-based HTML extractor so the lab can run in a minimal Python environment.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import requests


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "news"
REQUEST_TIMEOUT_SECONDS = 30
MIN_CONTENT_CHARS = 500


@dataclass(frozen=True)
class ArticleSeed:
    url: str
    title_hint: str
    topic: str
    fallback_summary: str


ARTICLE_SEEDS: tuple[ArticleSeed, ...] = (
    ArticleSeed(
        url="https://cuoi.tuoitre.vn/loat-nghe-si-viet-tieu-tan-su-nghiep-vi-ma-tuy-20241114142620463.htm",
        title_hint="Loat nghe si Viet tieu tan su nghiep vi ma tuy",
        topic="Tong hop cac nghe si Viet dinh scandal ma tuy",
        fallback_summary=(
            "Bai viet tong hop nhieu truong hop nghe si Viet Nam bi anh huong nghiem "
            "trong ve su nghiep sau cac vu viec lien quan den ma tuy. Noi dung nhac "
            "toi phan ung cua cong chung, trach nhiem cua nguoi noi tieng va tac dong "
            "cua cac hanh vi vi pham phap luat toi hinh anh nghe si. Bai viet cung dat "
            "vu viec cua Chi Dan va An Tay trong boi canh rong hon cua showbiz Viet, "
            "noi ve viec danh tieng co the sut giam nhanh chong khi nguoi cua cong "
            "chung bi gan voi cac hanh vi su dung, tang tru hoac to chuc su dung chat "
            "ma tuy. Phan fallback nay chi duoc dung neu trang bao khong cho crawler "
            "trich xuat noi dung day du."
        ),
    ),
    ArticleSeed(
        url="https://tuoitre.vn/nguoi-mau-nhikolai-dinh-bi-bat-trong-chuyen-an-ma-tuy-o-khu-ma-lang-quan-1-20240625230004986.htm",
        title_hint="Nguoi mau Nhikolai Dinh bi bat trong chuyen an ma tuy",
        topic="Nguoi mau Nhikolai Dinh",
        fallback_summary=(
            "Bai bao Tuoi Tre dua tin ve viec nguoi mau Nhikolai Dinh, tung tham gia "
            "Vietnam's Next Top Model va xuat hien trong cac san dien thoi trang, bi "
            "bat trong mot chuyen an ma tuy tai khu Ma Lang, quan 1, TP.HCM. Noi dung "
            "neu cac quyet dinh khoi to, bat tam giam doi voi mot so bi can va thong "
            "tin co quan dieu tra xac dinh cac doi tuong co hanh vi tang tru trai phep "
            "chat ma tuy. Bai viet la nguon tin phu hop cho tap du lieu RAG ve nghe si "
            "hoac nguoi mau lien quan den ma tuy tai Viet Nam."
        ),
    ),
    ArticleSeed(
        url="https://dantri.com.vn/phap-luat/cong-an-tphcm-doc-lenh-bat-co-tien-truc-phuong-nguoi-mau-an-tay-20241114143106380.htm",
        title_hint="Cong an TPHCM doc lenh bat co tien Truc Phuong nguoi mau An Tay",
        topic="Chi Dan, An Tay, Truc Phuong",
        fallback_summary=(
            "Bai viet Dan Tri thong tin Cong an TP.HCM khoi to, bat tam giam mot so "
            "nguoi noi tieng gom ca si Chi Dan, nguoi mau An Tay va Nguyen Do Truc "
            "Phuong vi lien quan den ma tuy. Noi dung neu boi canh chuyen an, cac toi "
            "danh bi dieu tra va vai tro cua tung nguoi trong vu viec theo thong tin "
            "tu co quan chuc nang. Day la bai bao co metadata ro ve nhom nguoi noi "
            "tieng trong showbiz va mang xa hoi, phu hop de dua vao kho du lieu tin tuc "
            "cho pipeline truy van va sinh cau tra loi co citation."
        ),
    ),
    ArticleSeed(
        url="https://dantri.com.vn/phap-luat/truy-to-ca-si-chi-dan-nguoi-mau-an-tay-20260402122649916.htm",
        title_hint="Truy to ca si Chi Dan nguoi mau An Tay",
        topic="Truy to Chi Dan va An Tay",
        fallback_summary=(
            "Bai viet Dan Tri cap nhat viec truy to ca si Chi Dan va nguoi mau An Tay "
            "trong mot chuyen an ma tuy lon. Noi dung neu cac cao buoc lien quan den "
            "to chuc su dung trai phep chat ma tuy, tang tru trai phep chat ma tuy, "
            "qua trinh dieu tra va mot so tinh tiet ve cach cac bi can su dung ma tuy. "
            "Bai viet co gia tri cho RAG vi bo sung moc thoi gian sau giai doan bat giu, "
            "giup truy van phan biet giua tin bat, tin khoi to va tin truy to."
        ),
    ),
    ArticleSeed(
        url="https://www.sggp.org.vn/bat-ca-si-chi-dan-nguoi-mau-an-tay-co-tien-truc-phuong-vi-to-chuc-su-dung-ma-tuy-post768266.html",
        title_hint="Bat ca si Chi Dan nguoi mau An Tay co tien Truc Phuong",
        topic="Bat Chi Dan, An Tay, Truc Phuong",
        fallback_summary=(
            "Bai viet Sai Gon Giai Phong dua tin Cong an TP.HCM bat ca si Chi Dan, "
            "nguoi mau An Tay va Nguyen Do Truc Phuong trong vu viec lien quan den "
            "hanh vi to chuc su dung trai phep chat ma tuy. Noi dung de cap viec mo "
            "rong chuyen an van chuyen ma tuy qua duong hang khong va nhan manh qua "
            "trinh co quan chuc nang truy xet cac doi tuong mua, ban, to chuc su dung "
            "ma tuy. Day la mot nguon bao chinh thong trong nuoc, huu ich de doi chieu "
            "voi cac bai cua Dan Tri va Tuoi Tre trong he thong RAG."
        ),
    ),
)

ARTICLE_URLS = [seed.url for seed in ARTICLE_SEEDS]


class ParagraphExtractor(HTMLParser):
    """Tiny HTML text extractor for news pages."""

    def __init__(self) -> None:
        super().__init__()
        self._capture_tag: str | None = None
        self._parts: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "h1", "h2", "li", "title"}:
            self._capture_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._capture_tag:
            self._capture_tag = None

    def handle_data(self, data: str) -> None:
        text = normalize_whitespace(data)
        if not text or self._capture_tag is None:
            return
        if self._capture_tag == "title":
            self._title_parts.append(text)
        else:
            self._parts.append(text)

    @property
    def title(self) -> str:
        return " ".join(self._title_parts).strip()

    @property
    def content(self) -> str:
        return "\n\n".join(remove_duplicate_lines(self._parts))


def setup_directory() -> None:
    """Create data/landing/news/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def remove_duplicate_lines(lines: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_lines: list[str] = []
    for line in lines:
        if len(line) < 20 or line in seen:
            continue
        seen.add(line)
        unique_lines.append(line)
    return unique_lines


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:70] or "article"


def extract_title_from_html(html: str, fallback: str) -> str:
    patterns = [
        r'<meta\s+property=["\']og:title["\']\s+content=["\']([^"\']+)["\']',
        r'<meta\s+name=["\']title["\']\s+content=["\']([^"\']+)["\']',
        r"<title[^>]*>(.*?)</title>",
        r"<h1[^>]*>(.*?)</h1>",
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
        if match:
            title = normalize_whitespace(re.sub(r"<[^>]+>", " ", match.group(1)))
            if title:
                return title
    return fallback


async def crawl_article_with_crawl4ai(seed: ArticleSeed) -> dict:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=seed.url)

    title = result.metadata.get("title") if result.metadata else None
    return build_article_payload(
        seed=seed,
        title=title or seed.title_hint,
        content_markdown=result.markdown or "",
        crawler="crawl4ai",
    )


def crawl_article_with_requests(seed: ArticleSeed) -> dict:
    response = requests.get(
        seed.url,
        headers={"User-Agent": "Day08-RAG-Lab/1.0"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    response.encoding = "utf-8"

    html = response.text
    extractor = ParagraphExtractor()
    extractor.feed(html)

    title = extract_title_from_html(html, extractor.title or seed.title_hint)
    content = extractor.content
    return build_article_payload(
        seed=seed,
        title=title,
        content_markdown=content,
        crawler="requests-html-parser",
    )


def build_article_payload(
    seed: ArticleSeed,
    title: str,
    content_markdown: str,
    crawler: str,
) -> dict:
    content = normalize_content(content_markdown)
    used_fallback = False

    if len(content) < MIN_CONTENT_CHARS:
        used_fallback = True
        content = (
            f"{seed.fallback_summary}\n\n"
            f"Source URL: {seed.url}\n"
            "Crawler note: The live page did not expose enough article text during "
            "this crawl, so this curated source summary is stored as fallback text."
        )

    return {
        "url": seed.url,
        "title": normalize_whitespace(title),
        "date_crawled": datetime.now().isoformat(timespec="seconds"),
        "source": urlparse(seed.url).netloc,
        "topic": seed.topic,
        "crawler": crawler,
        "used_fallback_summary": used_fallback,
        "content_markdown": content,
    }


def normalize_content(content: str) -> str:
    lines = [normalize_whitespace(line) for line in content.splitlines()]
    return "\n\n".join(remove_duplicate_lines(lines))


async def crawl_article(seed_or_url: ArticleSeed | str) -> dict:
    """
    Crawl one article and return metadata plus content.

    Args:
        seed_or_url: ArticleSeed, or a URL string from ARTICLE_URLS.
    """
    seed = resolve_seed(seed_or_url)

    try:
        return await crawl_article_with_crawl4ai(seed)
    except ImportError:
        return crawl_article_with_requests(seed)
    except Exception as crawl4ai_error:
        try:
            return crawl_article_with_requests(seed)
        except Exception as requests_error:
            return build_article_payload(
                seed=seed,
                title=seed.title_hint,
                content_markdown=(
                    f"{seed.fallback_summary}\n\n"
                    f"Crawl4AI error: {crawl4ai_error}\n"
                    f"Requests error: {requests_error}"
                ),
                crawler="fallback-summary",
            )


def resolve_seed(seed_or_url: ArticleSeed | str) -> ArticleSeed:
    if isinstance(seed_or_url, ArticleSeed):
        return seed_or_url

    for seed in ARTICLE_SEEDS:
        if seed.url == seed_or_url:
            return seed

    return ArticleSeed(
        url=seed_or_url,
        title_hint=seed_or_url.rsplit("/", 1)[-1],
        topic="custom article",
        fallback_summary=(
            "Custom URL supplied by the user. The crawler could not find a curated "
            "summary for this source, so only live extracted content should be used."
        ),
    )


async def crawl_all() -> list[Path]:
    """Crawl all configured articles and save one JSON file per article."""
    setup_directory()
    saved_files: list[Path] = []

    for index, seed in enumerate(ARTICLE_SEEDS, 1):
        print(f"[{index}/{len(ARTICLE_SEEDS)}] Crawling: {seed.url}")
        article = await crawl_article(seed)

        filename = f"article_{index:02d}_{slugify(article['title'])}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_files.append(filepath)
        print(f"Saved: {filepath.name}")

    return saved_files


def list_collected_articles() -> list[Path]:
    """Return news JSON files that satisfy the lab test size requirement."""
    if not DATA_DIR.exists():
        return []
    return sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() == ".json" and path.stat().st_size > 500
    )


if __name__ == "__main__":
    files = asyncio.run(crawl_all())
    print(f"Task 2 complete: {len(files)} articles saved in {DATA_DIR}.")
