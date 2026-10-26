"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields

from dotenv import load_dotenv


def _coerce(raw: str, default: object) -> object:
    if isinstance(default, bool):
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass(frozen=True)
class Settings:
    """Every knob of the pipeline. Set any field with the env var INVOICEIQ_<FIELD_NAME>.

    The Gemini key itself is read by the Google SDK from GEMINI_API_KEY.
    """

    extractor: str = "gemini"  # gemini | ocr-rules | donut
    gemini_model: str = "gemini-3.5-flash-lite"
    # Models to switch to when the main one runs out of free quota or is overloaded
    gemini_fallback_models: str = "gemini-3.1-flash-lite,gemini-3.6-flash,gemini-3.8-flash"
    thinking_level: str = "low"  # Gemini 3 models: minimal | low | medium | high
    check_printed: bool = True  # confirm every amount against local OCR of the page
    tolerance: float = 0.005  # relative tolerance of the arithmetic checks
    data_dir: str = "data"  # downloaded benchmark data and the OCR cache
    results_dir: str = "results"  # benchmark predictions and metrics

    @property
    def fallback_models(self) -> list[str]:
        return [m.strip() for m in self.gemini_fallback_models.split(",") if m.strip()]

    @classmethod
    def from_env(cls, **overrides: object) -> Settings:
        """Build settings from the environment; keyword overrides that are not None win."""
        load_dotenv()
        values: dict[str, object] = {}
        for field in fields(cls):
            raw = os.getenv(f"INVOICEIQ_{field.name.upper()}")
            if raw not in (None, ""):
                values[field.name] = _coerce(raw, field.default)
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls(**values)
