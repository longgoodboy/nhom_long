"""
Task 3 - Convert landing files into Markdown.

Input:
    data/landing/legal/*.pdf|*.doc|*.docx
    data/landing/news/*.json

Output:
    data/standardized/legal/*.md
    data/standardized/news/*.md

MarkItDown is used for legal PDF/DOC files when available. News JSON files are
already crawled with markdown-like article content, so they are converted with a
small structured formatter.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent.parent
LANDING_DIR = PROJECT_DIR / "data" / "landing"
OUTPUT_DIR = PROJECT_DIR / "data" / "standardized"
LEGAL_INPUT_DIR = LANDING_DIR / "legal"
NEWS_INPUT_DIR = LANDING_DIR / "news"
LEGAL_OUTPUT_DIR = OUTPUT_DIR / "legal"
NEWS_OUTPUT_DIR = OUTPUT_DIR / "news"

SUPPORTED_LEGAL_EXTENSIONS = {".pdf", ".doc", ".docx"}


def get_markitdown_converter():
    """Return a MarkItDown converter instance, or None if it is not installed."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        return None
    return MarkItDown()


def setup_directories() -> None:
    """Create standardized output directories."""
    LEGAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:90] or "document"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_legal_manifest() -> dict[str, dict[str, Any]]:
    manifest_path = LEGAL_INPUT_DIR / "manifest.json"
    if not manifest_path.exists():
        return {}

    manifest = read_json(manifest_path)
    documents = manifest.get("documents", [])
    return {
        item.get("filename", ""): item
        for item in documents
        if isinstance(item, dict) and item.get("filename")
    }


def convert_legal_file(filepath: Path, converter: Any, metadata: dict[str, Any]) -> str:
    """Convert one legal PDF/DOC file into markdown text."""
    title = metadata.get("title") or filepath.stem.replace("-", " ").title()
    document_number = metadata.get("document_number", "N/A")
    issued_date = metadata.get("issued_date", "N/A")
    source_page = metadata.get("source_page", "N/A")

    header = (
        f"# {title}\n\n"
        f"**Document number:** {document_number}\n\n"
        f"**Issued date:** {issued_date}\n\n"
        f"**Source:** {source_page}\n\n"
        f"**Original file:** {filepath.name}\n\n"
        "---\n\n"
    )

    if converter is None:
        body = (
            "MarkItDown is not installed in this Python environment, so the original "
            "document was registered with source metadata only. Install dependencies "
            "with `pip install -r requirements.txt` and rerun this script to extract "
            "full text from the PDF/DOC file.\n"
        )
        return header + body

    try:
        result = converter.convert(str(filepath))
        body = getattr(result, "text_content", "") or ""
    except Exception as exc:
        body = (
            "MarkItDown could not extract this document during conversion.\n\n"
            f"Conversion error: `{exc}`\n\n"
            "The original file remains available in `data/landing/legal/`.\n"
        )

    if len(body.strip()) < 100:
        body += (
            "\n\nConversion produced very little text. Keep the original file as the "
            "source of truth and rerun conversion after installing PDF dependencies.\n"
        )

    return header + body.strip() + "\n"


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOCX files in data/landing/legal/ into markdown."""
    LEGAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    converter = get_markitdown_converter()
    manifest_by_filename = load_legal_manifest()
    output_files: list[Path] = []

    if not LEGAL_INPUT_DIR.exists():
        print(f"Legal input directory does not exist: {LEGAL_INPUT_DIR}")
        return output_files

    for filepath in sorted(LEGAL_INPUT_DIR.iterdir()):
        if filepath.suffix.lower() not in SUPPORTED_LEGAL_EXTENSIONS:
            continue

        print(f"Converting legal document: {filepath.name}")
        metadata = manifest_by_filename.get(filepath.name, {})
        markdown = convert_legal_file(filepath, converter, metadata)

        output_path = LEGAL_OUTPUT_DIR / f"{filepath.stem}.md"
        output_path.write_text(markdown, encoding="utf-8")
        output_files.append(output_path)
        print(f"Saved: {output_path}")

    return output_files


def format_news_article(data: dict[str, Any], source_filename: str) -> str:
    """Format one crawled JSON article as a markdown document."""
    title = data.get("title") or source_filename
    url = data.get("url", "N/A")
    date_crawled = data.get("date_crawled", "N/A")
    source = data.get("source", "N/A")
    topic = data.get("topic", "N/A")
    crawler = data.get("crawler", "N/A")
    content = data.get("content_markdown") or data.get("content") or ""

    header = (
        f"# {title}\n\n"
        f"**Source:** {source}\n\n"
        f"**URL:** {url}\n\n"
        f"**Crawled:** {date_crawled}\n\n"
        f"**Topic:** {topic}\n\n"
        f"**Crawler:** {crawler}\n\n"
        f"**Original file:** {source_filename}\n\n"
        "---\n\n"
    )

    if not str(content).strip():
        content = "No article content was found in the crawled JSON file."

    return header + str(content).strip() + "\n"


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON articles in data/landing/news/ into markdown."""
    NEWS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_files: list[Path] = []

    if not NEWS_INPUT_DIR.exists():
        print(f"News input directory does not exist: {NEWS_INPUT_DIR}")
        return output_files

    for filepath in sorted(NEWS_INPUT_DIR.iterdir()):
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting news article: {filepath.name}")
        data = read_json(filepath)
        markdown = format_news_article(data, filepath.name)

        title = data.get("title") or filepath.stem
        output_path = NEWS_OUTPUT_DIR / f"{slugify(title)}.md"
        output_path.write_text(markdown, encoding="utf-8")
        output_files.append(output_path)
        print(f"Saved: {output_path}")

    return output_files


def convert_all() -> dict[str, list[Path]]:
    """Convert all supported landing files into standardized markdown."""
    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)
    setup_directories()

    print("\n--- Legal Documents ---")
    legal_files = convert_legal_docs()

    print("\n--- News Articles ---")
    news_files = convert_news_articles()

    print(f"\nDone. Legal markdown files: {len(legal_files)}")
    print(f"Done. News markdown files: {len(news_files)}")
    print(f"Output directory: {OUTPUT_DIR}")
    return {"legal": legal_files, "news": news_files}


if __name__ == "__main__":
    convert_all()
