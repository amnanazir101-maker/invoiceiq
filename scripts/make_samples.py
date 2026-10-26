"""Save Gemini's benchmark results for the sample receipts, so the demo can replay them.

The three files in examples/ are CORD test receipts. Their Gemini results come from
results/gemini-test/predictions.jsonl; the OCR text for the printed-amounts check is read from
the example images themselves. Writes examples/samples.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from invoiceiq.bench import read_records
from invoiceiq.documents import Document
from invoiceiq.ocr import OcrEngine

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = {  # example file -> CORD test id (see examples/README.md)
    "receipt-add-ons.jpg": "test-016",
    "receipt-faded.jpg": "test-020",
    "receipt-two-line-items.jpg": "test-026",
}


def main() -> None:
    records = {
        r["id"]: r for r in read_records(ROOT / "results" / "gemini-test" / "predictions.jsonl")
    }
    ocr = OcrEngine()
    saved = {}
    for name, cord_id in SAMPLES.items():
        record = records[cord_id]
        document = Document.from_path(ROOT / "examples" / name)
        saved[name] = {
            "cord_id": cord_id,
            "model": record["model"],
            "seconds": record["seconds"],
            "receipt": record["receipt"],
            "ocr_text": ocr.read_pages(document.pages),
        }
        print(f"{name}: {record['model']}, {record['status']}")
    out = ROOT / "examples" / "samples.json"
    out.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
