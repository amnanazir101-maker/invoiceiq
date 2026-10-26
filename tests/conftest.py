import pytest

from invoiceiq.documents import Document
from invoiceiq.extractors.base import Extraction
from invoiceiq.schema import LineItem, Receipt


class FakeExtractor:
    """Returns a fixed receipt, so pipelines can be tested without a model."""

    name = "fake"

    def __init__(self, receipt: Receipt | None = None) -> None:
        self.receipt = receipt or Receipt(
            items=[
                LineItem(name="Kopi Susu", quantity="2", unit_price="@9,000", price="18,000"),
                LineItem(name="Roti Bakar", quantity="1", price="30,000"),
            ],
            subtotal="48,000",
            tax="4,800",
            total="52,800",
        )
        self.calls = 0

    def extract(self, document: Document) -> Extraction:
        self.calls += 1
        return Extraction(
            receipt=self.receipt.model_copy(deep=True), model="fake-model", seconds=0.01
        )


@pytest.fixture
def fake_extractor() -> FakeExtractor:
    return FakeExtractor()
