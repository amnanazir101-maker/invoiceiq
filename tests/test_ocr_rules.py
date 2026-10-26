import pytest
from PIL import Image, ImageDraw, ImageFont

from invoiceiq.documents import Document
from invoiceiq.extractors.ocr_rules import OcrRulesExtractor, parse_lines
from invoiceiq.ocr import OcrPage, TextBox, group_lines


def test_parse_lines_reads_items_and_summary():
    receipt = parse_lines(
        [
            "1X M-Caramel Black Tea @28,000 28,000",
            "Less Ice",
            "2 Roti Bakar 30.000",
            "Subtotal: 58,000",
            "PB1 10% 5,800",
            "Total: 63,800",
            "Tunai 100.000",
            "Kembali 36.200",
        ]
    )
    first, second = receipt.items
    assert (first.quantity, first.name, first.unit_price, first.price) == (
        "1X", "M-Caramel Black Tea", "@28,000", "28,000",
    )  # fmt: skip
    assert [sub.name for sub in first.sub_items] == ["Less Ice"]
    assert (second.quantity, second.name, second.price) == ("2", "Roti Bakar", "30.000")
    assert (receipt.subtotal, receipt.tax, receipt.total) == ("58,000", "5,800", "63,800")
    assert (receipt.cash, receipt.change) == ("100.000", "36.200")


def test_an_amount_on_the_next_line_belongs_to_the_label():
    receipt = parse_lines(["Kopi 9.000", "TOTAL", "9.000"])
    assert receipt.total == "9.000"


def test_a_column_header_does_not_end_the_items():
    receipt = parse_lines(["QTY ITEM PRICE", "1 Es Teh 5.000", "TOTAL 5.000"])
    assert [item.name for item in receipt.items] == ["Es Teh"]


def test_dates_times_and_phone_numbers_are_not_money():
    receipt = parse_lines(["Telp 0218765432", "25/10/2026 12:30", "Kopi 9.000", "Total 9.000"])
    assert [item.name for item in receipt.items] == ["Kopi"]


def test_group_lines_joins_name_and_price_boxes():
    boxes = [
        TextBox("TOTAL", 10, 100, 80, 120, 0.99),
        TextBox("52,800", 300, 102, 380, 121, 0.99),
        TextBox("KOPI", 10, 50, 60, 70, 0.99),
        TextBox("18,000", 300, 49, 380, 68, 0.99),
    ]
    assert OcrPage(boxes).text == "KOPI 18,000\nTOTAL 52,800"
    assert len(group_lines(boxes)) == 2


@pytest.fixture(scope="module")
def receipt_image() -> Image.Image:
    image = Image.new("RGB", (520, 330), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 22)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except OSError:
            font = ImageFont.load_default(size=22)
    rows = [("KOPI SUSU", "18,000"), ("ROTI BAKAR", "30,000"), ("SUBTOTAL", "48,000"),
            ("TOTAL", "48,000"), ("CASH", "50,000"), ("CHANGE", "2,000")]  # fmt: skip
    for row, (label, amount) in enumerate(rows):
        draw.text((24, 20 + row * 48), label, fill="black", font=font)
        draw.text((380, 20 + row * 48), amount, fill="black", font=font)
    return image


def test_real_ocr_reads_a_rendered_receipt(receipt_image):
    extraction = OcrRulesExtractor().extract(Document.from_image(receipt_image))
    receipt = extraction.receipt
    assert [item.price for item in receipt.items] == ["18,000", "30,000"]
    assert (receipt.subtotal, receipt.total, receipt.cash, receipt.change) == (
        "48,000", "48,000", "50,000", "2,000",
    )  # fmt: skip
    assert "KOPI SUSU" in extraction.ocr_text
