"""Task 1 - Collect legal documents for the individual RAG pipeline.

The actual PDF files are stored in data/landing/legal.  This module keeps the
collection step reproducible by exposing helpers to create the directory,
inspect collected files, and validate that the minimum corpus is present.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
VALID_EXTENSIONS = {".pdf", ".docx", ".doc"}
MIN_LEGAL_FILES = 3


def setup_directory() -> Path:
    """Create data/landing/legal and return its path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def list_legal_documents() -> list[Path]:
    """Return collected legal source files sorted by name."""
    setup_directory()
    return sorted(
        path for path in DATA_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def validate_collection(min_files: int = MIN_LEGAL_FILES) -> dict:
    """Validate the local legal corpus used by the tests and demo."""
    files = list_legal_documents()
    non_empty = [path for path in files if path.stat().st_size > 1024]
    return {
        "directory": str(DATA_DIR),
        "file_count": len(files),
        "non_empty_count": len(non_empty),
        "ok": len(files) >= min_files and len(non_empty) >= min_files,
        "files": [path.name for path in files],
    }


if __name__ == "__main__":
    report = validate_collection()
    print(f"Legal data directory: {report['directory']}")
    print(f"Files: {report['file_count']} collected, {report['non_empty_count']} non-empty")
    for name in report["files"]:
        print(f"- {name}")
    if not report["ok"]:
        raise SystemExit("Need at least 3 non-empty legal PDF/DOC/DOCX files.")
