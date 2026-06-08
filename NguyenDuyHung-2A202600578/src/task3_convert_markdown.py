"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    pip install "markitdown[all]"

Flow:
    1. Scan file trong data/landing/legal + news
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc
"""

import json
from pathlib import Path

from markitdown import MarkItDown
from markitdown._exceptions import FileConversionException, MissingDependencyException


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =========================
# INIT MARKITDOWN
# =========================
md = MarkItDown()


# =========================
# LEGAL DOCS CONVERSION
# =========================
def convert_legal_docs():
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("[WARN] legal dir not found")
        return

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() not in [".pdf", ".docx", ".doc"]:
            continue

        print(f"Converting: {filepath.name}")

        try:
            result = md.convert(str(filepath))
            text = getattr(result, "text_content", "")

            # remove whitespace noise
            text = text.strip()

            # ❌ skip invalid output
            if len(text) < 50:
                print(f"  ✗ Skip (empty/too short): {filepath.name}")
                continue

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text, encoding="utf-8")

            print(f"  ✓ Saved: {output_path}")

        except (FileConversionException, MissingDependencyException) as e:
            print(f"  ✗ Convert failed: {filepath.name} -> {e}")
            continue

        except Exception as e:
            print(f"  ✗ Unexpected error: {filepath.name} -> {e}")
            continue


# =========================
# NEWS JSON CONVERSION
# =========================
def convert_news_articles():
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("[WARN] news dir not found")
        return

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))

            title = data.get("title", "Unknown")
            url = data.get("url", "N/A")
            date = data.get("date_crawled", "N/A")
            content = data.get("content_markdown", "")

            content = content.strip()

            # skip empty content
            if len(content) < 50:
                print(f"  ✗ Skip empty news: {filepath.name}")
                continue

            md_content = (
                f"# {title}\n\n"
                f"**Source:** {url}\n"
                f"**Crawled:** {date}\n\n"
                f"---\n\n"
                f"{content}"
            )

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(md_content, encoding="utf-8")

            print(f"  ✓ Saved: {output_path}")

        except Exception as e:
            print(f"  ✗ Failed JSON: {filepath.name} -> {e}")
            continue


# =========================
# MAIN PIPELINE
# =========================
def convert_all():
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown FIXED)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()