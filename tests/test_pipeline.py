import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from invoiceiq import bench
from invoiceiq.api import create_app
from invoiceiq.cord import Sample
from invoiceiq.documents import Document, DocumentError, jpeg_bytes
from invoiceiq.export import to_excel, to_json
from invoiceiq.extractors.base import ExtractionError, RateLimitError
from invoiceiq.pipeline import Pipeline


class FakeOcr:
    def __init__(self, text: str) -> None:
        self.text = text

    def read_pages(self, pages) -> str:
        return self.text


def png_bytes(size=(300, 400)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def document() -> Document:
    return Document.from_bytes(png_bytes(), "receipt.png")


def test_pipeline_extracts_and_validates(fake_extractor, document):
    result = Pipeline(fake_extractor).process(document)
    assert result.review.status == "approved"
    assert result.values["total"] == 52800.0
    assert result.to_dict()["status"] == "approved"


def test_ocr_text_enables_the_printed_amounts_check(fake_extractor, document):
    ocr = FakeOcr(
        "KOPI SUSU 2 @9,000 18,000\nROTI BAKAR 30,000\nSUBTOTAL 48,000\nTAX 4,800\nTOTAL 58,800"
    )
    result = Pipeline(fake_extractor, ocr).process(document)
    assert result.review.status == "needs_review"
    assert result.review.flagged == ["total"]  # 52,800 is not printed anywhere


def test_a_corrected_field_passes_the_checks_again(fake_extractor, document):
    pipeline = Pipeline(fake_extractor)
    result = pipeline.process(document)
    broken = result.receipt.model_copy(update={"total": "58,800"})
    assert pipeline.review(broken, None).status == "needs_review"
    fixed = broken.model_copy(update={"total": "52,800"})
    assert pipeline.review(fixed, None).status == "approved"


def test_documents_accept_pdfs_and_reject_other_files():
    buffer = io.BytesIO()
    Image.new("RGB", (400, 600), "white").save(buffer, format="PDF")
    pdf = Document.from_bytes(buffer.getvalue(), "invoice.pdf")
    assert len(pdf.pages) == 1 and pdf.pages[0].mode == "RGB"
    with pytest.raises(DocumentError):
        Document.from_bytes(b"not an image", "notes.txt")


def test_upload_copies_are_downscaled():
    image = Image.open(io.BytesIO(jpeg_bytes(Image.new("RGB", (3024, 4032)), max_side=2048)))
    assert max(image.size) == 2048


def test_excel_export_has_receipts_items_and_checks(fake_extractor, document):
    result = Pipeline(fake_extractor).process(document)
    sheets = pd.read_excel(io.BytesIO(to_excel([result])), sheet_name=None)
    assert list(sheets) == ["Receipts", "Items", "Checks"]
    assert sheets["Receipts"].loc[0, "total"] == 52800
    assert sheets["Items"]["price"].tolist() == [18000, 30000]
    assert json.loads(to_json([result]))["receipt"]["total"] == "52,800"


def test_api_returns_validated_json(fake_extractor):
    client = TestClient(create_app(Pipeline(fake_extractor)))
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/v1/extract", files={"file": ("receipt.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "approved"
    assert body["values"]["subtotal"] == 48000.0
    assert body["review"]["flagged"] == []


def test_api_rejects_files_it_cannot_read(fake_extractor):
    client = TestClient(create_app(Pipeline(fake_extractor)))
    response = client.post("/v1/extract", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


def test_api_reports_rate_limits(fake_extractor):
    class Limited:
        name = "limited"

        def extract(self, document):
            raise RateLimitError("Gemini's free-tier limit was reached.")

    client = TestClient(create_app(Pipeline(Limited())))
    response = client.post("/v1/extract", files={"file": ("r.png", png_bytes(), "image/png")})
    assert response.status_code == 429


def samples(n: int = 3) -> list[Sample]:
    gt = {"menu": [{"nm": "Kopi Susu", "cnt": "2", "unitprice": "@9,000", "price": "18,000"},
                   {"nm": "Roti Bakar", "cnt": "1", "price": "30,000"}],
          "sub_total": {"subtotal_price": "48,000", "tax_price": "4,800"},
          "total": {"total_price": "52,800"}}  # fmt: skip
    return [Sample(f"test-{i:03d}", png_bytes((60, 80)), gt) for i in range(n)]


def test_benchmark_scores_and_resumes(fake_extractor, tmp_path):
    pipeline = Pipeline(fake_extractor)
    records, stopped = bench.run(pipeline, samples(2), tmp_path)
    assert stopped is None and len(records) == 2
    records, _ = bench.run(pipeline, samples(3), tmp_path)  # only the new receipt is read
    assert fake_extractor.calls == 3 and len(records) == 3
    metrics = bench.summarize(records, samples(3), "fake", "test")
    assert metrics["field_f1"] == 1.0
    assert metrics["total_accuracy"] == 1.0
    assert metrics["auto_approved"] == 1.0
    assert metrics["fields"]["menu.nm"]["strict"]["support"] == 6
    bench.write_metrics(tmp_path / "fake-test", metrics)
    assert bench.load_metrics(tmp_path)[0]["run"] == "fake-test"


def test_benchmark_stops_on_rate_limits_and_retries_errors(fake_extractor, tmp_path):
    outcomes = [ExtractionError("overloaded"), RateLimitError("daily limit")]

    class Flaky:
        name = "flaky"

        def extract(self, document):
            if outcomes:
                raise outcomes.pop(0)
            return fake_extractor.extract(document)

    records, stopped = bench.run(Pipeline(Flaky()), samples(3), tmp_path)
    assert stopped == "daily limit"
    assert [r["id"] for r in records] == ["test-000"] and "error" in records[0]
    records, stopped = bench.run(Pipeline(Flaky()), samples(3), tmp_path)
    assert stopped is None
    assert all("error" not in r for r in records) and len(records) == 3


def test_markdown_tables_for_the_readme(fake_extractor, tmp_path):
    records, _ = bench.run(Pipeline(fake_extractor), samples(2), tmp_path)
    metrics = bench.summarize(records, samples(2), "ocr-rules", "test")
    table = bench.markdown([metrics])
    assert "| OCR + rules | 2 | 100.0% |" in table
    assert "Wrong receipts sent to review" in table


def test_labels_name_the_model():
    assert bench.label(
        {"extractor": "gemini", "models": {"gemini-3.8-flash": 99, "gemini-3.6-flash": 1}}
    ) == ("Gemini 3.8 Flash, zero-shot")
    assert bench.label({"extractor": "ocr-rules"}) == "OCR + rules"
