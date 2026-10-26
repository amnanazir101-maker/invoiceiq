import json
from pathlib import Path

import pandas as pd

from invoiceiq.cli import expand, main

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "examples" / "receipt-add-ons.jpg"


def test_wildcards_are_expanded_for_windows_shells():
    files = expand([ROOT / "examples" / "*.jpg"])
    assert SAMPLE in files and len(files) == 3


def test_extract_writes_json_and_excel_with_the_local_reader(tmp_path):
    out_json, out_excel = tmp_path / "out.json", tmp_path / "out.xlsx"
    code = main(
        ["extract", str(SAMPLE), "--extractor", "ocr-rules", "--json", str(out_json),
         "--excel", str(out_excel)]
    )  # fmt: skip
    assert code == 0
    result = json.loads(out_json.read_text(encoding="utf-8"))
    assert result["file"] == "receipt-add-ons.jpg"
    assert result["receipt"]["total"] == "70,000"
    assert result["values"]["total"] == 70000.0
    assert list(pd.read_excel(out_excel, sheet_name=None)) == ["Receipts", "Items", "Checks"]


def test_a_missing_file_is_reported(tmp_path):
    assert main(["extract", str(tmp_path / "nope.jpg"), "--extractor", "ocr-rules"]) == 1
