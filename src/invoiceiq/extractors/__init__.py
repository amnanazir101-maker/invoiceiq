"""Three ways to read a receipt, behind one interface, so they can be benchmarked side by side.

- `gemini`: a vision-language model with structured output, zero-shot (free tier).
- `donut`: an OCR-free model fine-tuned on CORD receipts, the specialist to beat (local CPU).
- `ocr-rules`: OCR followed by keyword rules, the classic approach (local CPU).
"""

from __future__ import annotations

from invoiceiq.config import Settings
from invoiceiq.extractors.base import Extraction, ExtractionError, Extractor, RateLimitError
from invoiceiq.ocr import OcrEngine

EXTRACTORS = ("gemini", "donut", "ocr-rules")


def make_extractor(
    name: str,
    settings: Settings,
    *,
    ocr: OcrEngine | None = None,
    model: str | None = None,
    fallback: bool = True,
) -> Extractor:
    if name == "gemini":
        from invoiceiq.extractors.gemini import GeminiExtractor

        return GeminiExtractor(
            model or settings.gemini_model,
            fallback_models=settings.fallback_models if fallback else (),
            thinking_level=settings.thinking_level,
        )
    if name == "donut":
        from invoiceiq.extractors.donut import DonutExtractor

        return DonutExtractor(model) if model else DonutExtractor()
    if name == "ocr-rules":
        from invoiceiq.extractors.ocr_rules import OcrRulesExtractor

        return OcrRulesExtractor(ocr)
    raise ValueError(f"Unknown extractor {name!r}; choose from {', '.join(EXTRACTORS)}.")


__all__ = [
    "EXTRACTORS", "Extraction", "ExtractionError", "Extractor", "RateLimitError", "make_extractor",
]  # fmt: skip
