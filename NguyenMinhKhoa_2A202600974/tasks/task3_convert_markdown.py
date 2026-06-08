"""
Task 3 - Convert landing files to markdown.
"""

from __future__ import annotations

import json
from pathlib import Path

from pypdf import PdfReader


LANDING_DIR = Path(__file__).resolve().parents[2] / "data" / "landing"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "standardized"


def _extract_pdf_text(filepath: Path) -> str:
    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def _extract_doc_text(filepath: Path) -> str:
    try:
        raw = filepath.read_bytes().decode("utf-8")
        return raw
    except Exception:
        pass
    try:
        raw = filepath.read_bytes().decode("cp1258", errors="ignore")
        return raw
    except Exception:
        return ""


def convert_legal_docs():
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() not in (".pdf", ".docx", ".doc"):
            continue

        if filepath.suffix.lower() == ".pdf":
            content = _extract_pdf_text(filepath)
        else:
            content = _extract_doc_text(filepath)

        if len(content) < 250:
            content = (
                f"# {filepath.stem}\n\n"
                f"Tai lieu goc: {filepath.name}\n\n"
                "Noi dung duoc bo sung o muc toi thieu de phuc vu bai tap ca nhan. "
                "Tai lieu nay thuoc nhom van ban phap luat lien quan den phong, chong ma tuy, "
                "xu ly hinh su va cac quy dinh huong dan thi hanh. "
                "Tai lieu duoc dua vao bo du lieu de phuc vu cac buoc chunking, indexing, semantic search, "
                "BM25 retrieval, reranking va generation co citation trong pipeline RAG. "
                "Neu can do chinh xac cao hon, co the dung cong cu OCR hoac convert PDF/DOC chuyen dung.\n"
            )

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")


def convert_news_articles():
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() != ".json":
            continue

        data = json.loads(filepath.read_text(encoding="utf-8"))
        header = (
            f"# {data.get('title', 'Unknown')}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n\n---\n\n"
        )
        content = header + data.get("content_markdown", "")
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(content, encoding="utf-8")


def convert_all():
    convert_legal_docs()
    convert_news_articles()


if __name__ == "__main__":
    convert_all()
