"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tải tối thiểu 3 văn bản pháp luật (PDF/DOC/DOCX) từ các nguồn chính thống
       và đặt trong data/landing/legal/. Đặt tên không dấu, có năm ban hành.
    2. Chạy script này để:
        - Convert mọi .doc (Word 97-2003) sang .docx (textutil) — MarkItDown
          ở task3 chỉ đọc được .docx.
        - PDF được giữ nguyên (để task8 upload lên PageIndex). Task3 cũng tự
          handle .pdf qua MarkItDown.
    3. Validate có đủ tối thiểu 3 file (.pdf hoặc .docx) trong thư mục.

Văn bản gợi ý:
    - Luật Phòng, chống ma tuý 2021 (73/2021/QH14)
    - Nghị định 105/2021/NĐ-CP
    - Bộ luật Hình sự 2015 (100/2015/QH13) — Chương XX
    - Luật Phòng, chống ma tuý 2025 (120/2025/QH15)

Nguồn tải:
    - https://thuvienphapluat.vn
    - https://vbpl.vn
    - https://vanban.chinhphu.vn
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

MIN_REQUIRED = 3
ACCEPTED_EXT = {".pdf", ".docx"}


def convert_doc_to_docx(src: Path) -> Path | None:
    """Convert .doc cũ (Word 97-2003) sang .docx qua macOS `textutil`."""
    dest = src.with_suffix(".docx")
    if dest.exists():
        return dest
    if not shutil.which("textutil"):
        print(f"  ⚠ Không có `textutil` (chỉ có trên macOS) — bỏ qua {src.name}")
        print(f"     Cài LibreOffice rồi chạy: soffice --convert-to docx {src}")
        return None
    subprocess.run(
        ["textutil", "-convert", "docx", "-output", str(dest), str(src)],
        check=True,
    )
    return dest


def normalize() -> list[Path]:
    """Convert mọi .doc trong DATA_DIR sang .docx. PDF giữ nguyên."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for src in list(DATA_DIR.iterdir()):
        if src.suffix.lower() == ".doc":
            print(f"→ Convert {src.name} (.doc → .docx)")
            convert_doc_to_docx(src)
    return sorted(f for f in DATA_DIR.iterdir() if f.suffix.lower() in ACCEPTED_EXT)


def validate(files: list[Path]) -> None:
    if len(files) < MIN_REQUIRED:
        raise SystemExit(
            f"✗ Cần tối thiểu {MIN_REQUIRED} file .pdf/.docx trong {DATA_DIR}, "
            f"hiện có {len(files)}. Vui lòng tải thêm văn bản pháp luật về."
        )
    print(f"\n✓ Có {len(files)} văn bản pháp luật trong landing zone:")
    for f in files:
        size_kb = f.stat().st_size // 1024
        print(f"  • {f.name} ({size_kb} KB)")


if __name__ == "__main__":
    print(f"Thư mục: {DATA_DIR}")
    files = normalize()
    validate(files)
