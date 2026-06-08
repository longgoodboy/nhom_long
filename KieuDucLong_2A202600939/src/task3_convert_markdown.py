"""Task 3 - convert landing files into standardized Markdown.

The news converter is schema-aware for the current article_*.json files and
preserves the original JSON data. Legal conversion uses MarkItDown when called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from markitdown import MarkItDown
except Exception:  # pragma: no cover - dependency may be unavailable in tests
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _article_to_markdown(data: dict[str, Any], fallback_id: str) -> str:
    title = data.get("title") or fallback_id
    publisher = data.get("publisher") or data.get("source") or "Không rõ nguồn"
    url = data.get("url") or ""
    published = data.get("published_date") or data.get("date") or "Không rõ ngày đăng"
    crawled = data.get("crawl_date") or data.get("date_crawled") or "Không rõ ngày crawl"
    topic = data.get("topic") or "Tin tức ma túy"
    summary = data.get("summary") or ""
    content = data.get("content") or data.get("markdown") or data.get("content_markdown") or summary
    parts = [
        f"# {title}",
        "",
        f"**Nguồn:** {publisher}",
        f"**URL:** {url}",
        f"**Ngày đăng:** {published}",
        f"**Ngày crawl:** {crawled}",
        f"**Chủ đề:** {topic}",
        "",
    ]
    if summary:
        parts += ["## Tóm tắt", "", str(summary).strip(), ""]
    parts += ["## Nội dung", "", str(content).strip(), ""]
    return "\n".join(parts)


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOCX files in data/landing/legal to markdown when MarkItDown is available."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    if MarkItDown is None:
        raise RuntimeError("markitdown is not available")
    md = MarkItDown()
    outputs: list[Path] = []
    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in {".pdf", ".docx", ".doc"}:
            result = md.convert(str(filepath))
            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(result.text_content, encoding="utf-8")
            outputs.append(output_path)
    return outputs


def convert_news_articles() -> list[Path]:
    """Convert current article_*.json news data into standardized markdown files."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for filepath in sorted(news_dir.glob("article_*.json")):
        data = _read_json(filepath)
        if not isinstance(data, dict):
            continue
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(_article_to_markdown(data, filepath.stem), encoding="utf-8")
        outputs.append(output_path)
    return outputs


def convert_all(convert_legal: bool = False) -> dict[str, list[Path]]:
    """Convert all supported landing files; legal conversion is opt-in to avoid slow PDF work."""
    converted = {"legal": [], "news": convert_news_articles()}
    if convert_legal:
        converted["legal"] = convert_legal_docs()
    return converted


if __name__ == "__main__":
    result = convert_all(convert_legal=False)
    print(f"Converted {len(result['news'])} news articles to {OUTPUT_DIR / 'news'}")
