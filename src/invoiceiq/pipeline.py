"""Extract, then validate: one call from an uploaded file to a reviewed result."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from invoiceiq.amounts import as_numbers
from invoiceiq.config import Settings
from invoiceiq.documents import Document
from invoiceiq.extractors import Extractor, make_extractor
from invoiceiq.ocr import OcrEngine
from invoiceiq.schema import Receipt
from invoiceiq.validation import DEFAULT_TOLERANCE, Review, validate


@dataclass
class Result:
    file: str
    receipt: Receipt  # values exactly as printed
    review: Review
    model: str
    seconds: float  # wall time, extraction and checks together
    native: dict | None = None
    ocr_text: str | None = field(default=None, repr=False)
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def values(self) -> dict:
        """The same data with amounts and quantities as numbers."""
        return as_numbers(self.receipt)

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "status": self.review.status,
            "receipt": self.receipt.model_dump(),
            "values": self.values,
            "review": self.review.model_dump(),
            "model": self.model,
            "seconds": self.seconds,
        }


class Pipeline:
    """Runs an extractor and the validator on a document.

    With an OCR engine, the page is read locally while the extractor works (for Gemini, while
    the request is in flight), so the printed-amounts check adds little to the wait.
    """

    def __init__(
        self,
        extractor: Extractor,
        ocr: OcrEngine | None = None,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> None:
        self.extractor = extractor
        self.ocr = ocr
        self.tolerance = tolerance

    @classmethod
    def from_settings(cls, settings: Settings, extractor: str | None = None, **kwargs) -> Pipeline:
        ocr = OcrEngine() if settings.check_printed or extractor == "ocr-rules" else None
        name = extractor or settings.extractor
        return cls(make_extractor(name, settings, ocr=ocr, **kwargs), ocr, settings.tolerance)

    def process(self, document: Document) -> Result:
        started = time.perf_counter()
        if self.ocr is not None and not getattr(self.extractor, "uses_ocr", False):
            with ThreadPoolExecutor(max_workers=1) as pool:
                reading = pool.submit(self.ocr.read_pages, document.pages)
                extraction = self.extractor.extract(document)
                ocr_text = reading.result()
        else:
            extraction = self.extractor.extract(document)
            ocr_text = extraction.ocr_text
        return Result(
            file=document.name,
            receipt=extraction.receipt,
            review=validate(extraction.receipt, ocr_text, self.tolerance),
            model=extraction.model,
            seconds=round(time.perf_counter() - started, 2),
            native=extraction.native,
            ocr_text=ocr_text,
            usage=extraction.usage,
        )

    def review(self, receipt: Receipt, ocr_text: str | None) -> Review:
        """Re-run the checks, for example after a reviewer corrects a field."""
        return validate(receipt, ocr_text, self.tolerance)
