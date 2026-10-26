"""Uploads as page images: photos and scans (PNG, JPEG, WebP, TIFF, BMP) or PDFs."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

PDF_DPI = 200
MAX_PAGES = 5  # receipts and invoices are short; this bounds the cost of a stray long PDF
SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".pdf")


class DocumentError(ValueError):
    """The upload is not an image or PDF that can be read."""


@dataclass
class Document:
    name: str
    pages: list[Image.Image] = field(repr=False)  # RGB, upright

    @classmethod
    def from_bytes(cls, data: bytes, name: str = "document") -> Document:
        if data[:5] == b"%PDF-":
            return cls(name, _pdf_pages(data))
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise DocumentError(f"{name} is not an image or PDF file.") from exc
        return cls(name, [_upright(image)])

    @classmethod
    def from_path(cls, path: str | Path) -> Document:
        path = Path(path)
        return cls.from_bytes(path.read_bytes(), path.name)

    @classmethod
    def from_image(cls, image: Image.Image, name: str = "image") -> Document:
        return cls(name, [_upright(image)])


def _upright(image: Image.Image) -> Image.Image:
    """Apply the camera's EXIF rotation and drop transparency."""
    return ImageOps.exif_transpose(image).convert("RGB")


def _pdf_pages(data: bytes) -> list[Image.Image]:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise DocumentError("The PDF could not be opened.") from exc
    try:
        pages = [
            pdf[index].render(scale=PDF_DPI / 72).to_pil().convert("RGB")
            for index in range(min(len(pdf), MAX_PAGES))
        ]
    finally:
        pdf.close()
    if not pages:
        raise DocumentError("The PDF has no pages.")
    return pages


def jpeg_bytes(image: Image.Image, max_side: int = 2048, quality: int = 90) -> bytes:
    """A compact copy for upload; 2048 px is more than vision models read receipts at."""
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
