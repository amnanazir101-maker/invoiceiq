"""What every extractor provides: a document in, a Receipt out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from invoiceiq.documents import Document
from invoiceiq.schema import Receipt


class ExtractionError(RuntimeError):
    """The extractor could not read the document."""


class RateLimitError(ExtractionError):
    """Every model hit its free-tier limit; retrying later will work."""


@dataclass
class Extraction:
    receipt: Receipt
    model: str
    seconds: float
    native: dict | None = None  # the extractor's own output, such as Donut's CORD tree
    ocr_text: str | None = None  # set by extractors that read the page with OCR anyway
    usage: dict[str, int] = field(default_factory=dict)  # tokens, for API-based extractors


class Extractor(Protocol):
    name: str

    def extract(self, document: Document) -> Extraction: ...
