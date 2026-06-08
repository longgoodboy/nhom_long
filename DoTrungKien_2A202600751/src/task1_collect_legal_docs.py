"""
Task 1 - Collect legal documents about drug prevention and prohibited substances.

This script downloads at least three official Vietnamese legal documents into:
    data/landing/legal/

The automated tests only require non-empty PDF/DOC/DOCX files in that folder, but
the script also records source metadata so the documents can be cited later in the
RAG pipeline.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import requests


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "manifest.json"
MIN_FILE_SIZE_BYTES = 1024
REQUEST_TIMEOUT_SECONDS = 30
PDF_LINK_PATTERN = re.compile(r'href=["\']([^"\']+\.pdf[^"\']*)["\']', re.IGNORECASE)


@dataclass(frozen=True)
class LegalDocument:
    title: str
    document_number: str
    issued_date: str
    source_page: str
    download_url: str
    filename: str


LEGAL_DOCUMENTS: tuple[LegalDocument, ...] = (
    LegalDocument(
        title="Luat Phong, chong ma tuy",
        document_number="73/2021/QH14",
        issued_date="2021-03-30",
        source_page="https://congbao.chinhphu.vn/van-ban/nghi-quyet-so-73-2021-qh14-33659.htm",
        download_url="https://congbao.chinhphu.vn/tai-ve-van-ban-so-73-2021-qh14-33659-35651?format=pdf",
        filename="luat-phong-chong-ma-tuy-2021.pdf",
    ),
    LegalDocument(
        title="Nghi dinh huong dan Luat Phong, chong ma tuy",
        document_number="105/2021/ND-CP",
        issued_date="2021-12-04",
        source_page="https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-105-2021-nd-cp-34944.htm",
        download_url="https://congbao.chinhphu.vn/tai-ve-van-ban-so-105-2021-nd-cp-34944-37821?format=pdf",
        filename="nghi-dinh-105-2021.pdf",
    ),
    LegalDocument(
        title="Nghi dinh quy dinh cac danh muc chat ma tuy va tien chat",
        document_number="57/2022/ND-CP",
        issued_date="2022-08-25",
        source_page="https://congbao.chinhphu.vn/van-ban/nghi-dinh-so-57-2022-nd-cp-37734.htm",
        download_url="https://congbao.chinhphu.vn/tai-ve-van-ban-so-57-2022-nd-cp-37734-41623?format=pdf",
        filename="nghi-dinh-57-2022.pdf",
    ),
)


def setup_directory() -> None:
    """Create data/landing/legal/ if it does not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def download_file(document: LegalDocument, force: bool = False) -> Path:
    """Download one legal document and return its local path."""
    setup_directory()
    destination = DATA_DIR / document.filename

    if destination.exists() and destination.stat().st_size > MIN_FILE_SIZE_BYTES and not force:
        print(f"Skip existing file: {destination.name}")
        return destination

    content = fetch_pdf_content(document.download_url, document.source_page)
    if len(content) <= MIN_FILE_SIZE_BYTES:
        raise ValueError(
            f"Downloaded file is too small for {document.filename}: {len(content)} bytes"
        )

    if destination.suffix.lower() == ".pdf" and not content.startswith(b"%PDF"):
        raise ValueError(f"Downloaded content is not a PDF: {document.filename}")

    destination.write_bytes(content)
    print(f"Downloaded: {destination.name} ({len(content):,} bytes)")
    return destination


def fetch_pdf_content(download_url: str, source_page: str) -> bytes:
    """Fetch PDF bytes, following Cong Bao HTML download pages when needed."""
    response = requests.get(
        download_url,
        headers={"User-Agent": "Day08-RAG-Lab/1.0"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    if response.content.startswith(b"%PDF"):
        return response.content

    html = response.text
    candidates = PDF_LINK_PATTERN.findall(html)

    if not candidates:
        source_response = requests.get(
            source_page,
            headers={"User-Agent": "Day08-RAG-Lab/1.0"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        source_response.raise_for_status()
        html = source_response.text
        candidates = PDF_LINK_PATTERN.findall(html)

    if not candidates:
        raise ValueError(f"Could not find a PDF link from {download_url}")

    pdf_url = urljoin(response.url, candidates[0].replace("&amp;", "&"))
    pdf_response = requests.get(
        pdf_url,
        headers={"User-Agent": "Day08-RAG-Lab/1.0"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    pdf_response.raise_for_status()
    return pdf_response.content


def write_manifest(documents: Iterable[LegalDocument]) -> Path:
    """Write source metadata for later conversion, indexing, and citation."""
    setup_directory()
    payload = {
        "description": "Official legal documents collected for Day 8 RAG lab - Task 1.",
        "documents": [asdict(document) for document in documents],
    }
    MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote metadata manifest: {MANIFEST_PATH.name}")
    return MANIFEST_PATH


def collect_legal_documents(force: bool = False) -> list[Path]:
    """Download all configured legal documents into data/landing/legal/."""
    downloaded_files = [download_file(document, force=force) for document in LEGAL_DOCUMENTS]
    write_manifest(LEGAL_DOCUMENTS)
    return downloaded_files


def list_collected_files() -> list[Path]:
    """Return collected legal document files that satisfy the lab requirement."""
    if not DATA_DIR.exists():
        return []

    valid_extensions = {".pdf", ".doc", ".docx"}
    return sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in valid_extensions
        and path.stat().st_size > MIN_FILE_SIZE_BYTES
    )


if __name__ == "__main__":
    files = collect_legal_documents()
    print(f"Task 1 complete: {len(files)} legal documents are available.")
