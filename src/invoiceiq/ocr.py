"""Local OCR with RapidOCR (PP-OCR models on ONNX Runtime): no API key and no system packages.

It serves two purposes: the classic OCR + rules baseline reads receipts from it, and the
validator uses it to confirm that every amount a vision model returns is printed on the page.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class TextBox:
    """One run of text found on the page, with its bounding box in pixels."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class OcrPage:
    boxes: list[TextBox]
    seconds: float = 0.0

    def lines(self) -> list[list[TextBox]]:
        """Boxes grouped into printed lines, each read left to right."""
        return group_lines(self.boxes)

    @property
    def text(self) -> str:
        return "\n".join(" ".join(box.text for box in line) for line in self.lines())


def group_lines(boxes: list[TextBox]) -> list[list[TextBox]]:
    """Group boxes whose vertical centres are within half a text height of each other.

    Receipts put the item name on the left and its price on the right, so OCR returns them as
    separate boxes; grouping puts them back on one line.
    """
    rows: list[list[TextBox]] = []
    for box in sorted(boxes, key=lambda b: b.center_y):
        if rows:
            row = rows[-1]
            center = median(b.center_y for b in row)
            height = median(b.height for b in row)
            if abs(box.center_y - center) <= 0.5 * min(height, box.height):
                row.append(box)
                continue
        rows.append([box])
    return [sorted(row, key=lambda b: b.x0) for row in rows]


class OcrEngine:
    """RapidOCR, loaded on first use and shared between threads.

    With a `cache_dir`, results are stored per image, so benchmarks can be re-run without
    reading every receipt again.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._engine = None
        self._lock = threading.Lock()

    def read(self, image: Image.Image) -> OcrPage:
        cached = self._cache_path(image)
        if cached is not None and cached.exists():
            data = json.loads(cached.read_text(encoding="utf-8"))
            return OcrPage([TextBox(**box) for box in data["boxes"]], data["seconds"])
        page = self._run(image)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            payload = {"boxes": [asdict(box) for box in page.boxes], "seconds": page.seconds}
            cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return page

    def read_pages(self, pages: list[Image.Image]) -> str:
        return "\n".join(self.read(page).text for page in pages)

    def _run(self, image: Image.Image) -> OcrPage:
        with self._lock:  # one ONNX session; it already uses every core
            if self._engine is None:
                from rapidocr import RapidOCR

                self._engine = RapidOCR(params={"Global.log_level": "warning"})
            started = time.perf_counter()
            output = self._engine(np.asarray(image.convert("RGB")))
            seconds = time.perf_counter() - started
        boxes = []
        if output.boxes is not None:
            for quad, text, score in zip(output.boxes, output.txts, output.scores, strict=True):
                xs, ys = [float(p[0]) for p in quad], [float(p[1]) for p in quad]
                boxes.append(TextBox(text, min(xs), min(ys), max(xs), max(ys), float(score)))
        return OcrPage(boxes, round(seconds, 3))

    def _cache_path(self, image: Image.Image) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha1(image.tobytes() + repr(image.size).encode()).hexdigest()[:20]
        return self.cache_dir / f"{digest}.json"
