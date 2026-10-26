"""The web app and its charts, run headlessly with Streamlit's test harness."""

import pytest

from invoiceiq.bench import label
from invoiceiq.charts import TOKENS, benchmark_chart, benchmark_frame, style

RUNS = [
    {"extractor": "gemini", "models": {"gemini-3.8-flash": 100}, "receipts": 100,
     "field_f1": 0.9, "field_f1_normalized": 0.93, "total_accuracy": 0.97, "amounts_exact": 0.8},
    {"extractor": "ocr-rules", "receipts": 100, "field_f1": 0.5,
     "field_f1_normalized": 0.6, "total_accuracy": 0.7, "amounts_exact": 0.2},
]  # fmt: skip


def test_benchmark_chart_highlights_gemini():
    frame = benchmark_frame(RUNS, label)
    assert set(frame["extractor"]) == {"Gemini 3.8 Flash, zero-shot", "OCR + rules"}
    assert frame[frame["highlight"]]["extractor"].unique().tolist() == [
        "Gemini 3.8 Flash, zero-shot"
    ]
    for theme in ("light", "dark"):
        spec = style(benchmark_chart(frame, TOKENS[theme]), TOKENS[theme]).to_dict()
        assert spec["facet"]["field"] == "metric"


def test_app_reads_the_sample_receipts_with_the_local_reader():
    testing = pytest.importorskip("streamlit.testing.v1")
    app = testing.AppTest.from_file("../app.py", default_timeout=300)
    app.run()
    assert not app.exception
    app.sidebar.radio[0].set_value("OCR + rules (local)").run()
    app.button[0].click().run()  # "Try the sample receipts"
    assert not app.exception
    documents = app.metric[0]
    assert documents.label == "Documents" and int(documents.value) >= 1
