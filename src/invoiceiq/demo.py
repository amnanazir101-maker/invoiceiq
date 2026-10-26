"""What a public demo needs: saved results for the sample receipts and a daily request cap.

A hosted demo runs on its owner's free Gemini quota. The sample receipts are replayed from
results Gemini produced earlier (with the checks re-run live), so browsing the demo costs
nothing, and live uploads are capped per visitor and per day.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from datetime import date
from pathlib import Path

from invoiceiq.pipeline import Result
from invoiceiq.schema import Receipt
from invoiceiq.validation import DEFAULT_TOLERANCE, validate


class DailyQuota:
    """A thread-safe counter of live requests that resets every day; limit 0 means no limit."""

    def __init__(self, limit: int, today: Callable[[], date] = date.today) -> None:
        self.limit = limit
        self._today = today
        self._day = today()
        self._used = 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        """Use one request if any are left today."""
        with self._lock:
            if self._today() != self._day:
                self._day, self._used = self._today(), 0
            if self.limit and self._used >= self.limit:
                return False
            self._used += 1
            return True

    @property
    def used(self) -> int:
        return self._used


def load_samples(path: str | Path) -> dict[str, dict]:
    """Saved results by file name, or {} when there are none."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def replay(name: str, saved: dict, tolerance: float = DEFAULT_TOLERANCE) -> Result:
    """A saved extraction, with the checks run again on it."""
    receipt = Receipt.model_validate(saved["receipt"])
    ocr_text = saved.get("ocr_text")
    return Result(
        file=name,
        receipt=receipt,
        review=validate(receipt, ocr_text, tolerance),
        model=saved["model"],
        seconds=saved["seconds"],
        ocr_text=ocr_text,
    )
