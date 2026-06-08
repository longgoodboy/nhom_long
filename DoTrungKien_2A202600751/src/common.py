"""Shared helpers for the Day 8 RAG lab."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
LANDING_DIR = DATA_DIR / "landing"
STANDARDIZED_DIR = DATA_DIR / "standardized"
INDEX_DIR = DATA_DIR / "index"

TOKEN_PATTERN = re.compile(r"[\wÀ-ỹ]+", flags=re.UNICODE)


def ensure_project_on_path() -> None:
    """Allow task files to run both as modules and as direct scripts."""
    project_path = str(PROJECT_DIR)
    if project_path not in sys.path:
        sys.path.insert(0, project_path)


def load_env() -> None:
    load_dotenv(PROJECT_DIR / ".env")


def env(name: str, default: str = "") -> str:
    load_env()
    return os.getenv(name, default)


def has_real_env(name: str) -> bool:
    value = env(name)
    return bool(value and "xxx" not in value.lower())


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def safe_preview(text: str, limit: int = 100) -> str:
    """Return a console-safe preview for Windows code pages."""
    return ascii(text[:limit])[1:-1]


def slugify(text: str, limit: int = 90) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit] or "document"
