from decimal import Decimal

import pytest

from invoiceiq.amounts import (
    as_numbers,
    canonical,
    find_amounts,
    normalize_text,
    parse_amount,
    parse_quantity,
)
from invoiceiq.schema import LineItem, Receipt


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("16.500", "16500"),  # Indonesian thousands
        ("16,500", "16500"),
        ("16,500.00", "16500"),
        ("16.500,00", "16500"),
        ("1,234,567", "1234567"),
        ("48982.68", "48982.68"),
        ("4.5", "4.5"),
        ("12,50", "12.5"),
        ("Rp. 111,000", "111000"),
        ("Rp 16.500,-", "16500"),  # ",-" means no cents, not a minus
        ("@11000", "11000"),
        ("-60.000", "-60000"),
        ("(5,000)", "-5000"),
        ("5.000-", "-5000"),
        ("-2 600", "-2600"),  # space-grouped thousands
        ("[145,900]", "145900"),
        ("60.000)", "60000"),
        ("Disc 10% 5.000", "5000"),  # the percentage is skipped
        ("0", "0"),
    ],
)
def test_parse_amount(text, expected):
    assert canonical(parse_amount(text)) == expected


@pytest.mark.parametrize("text", [None, "", "free", "10%"])
def test_parse_amount_without_a_number(text):
    assert parse_amount(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [("2", 2), ("1X", 1), ("1 x", 1), ("x2", 2), ("1.00xITEMs", 1), ("(2", 2), ("0.25 kg", 0.25)],
)
def test_parse_quantity(text, expected):
    assert parse_quantity(text) == Decimal(str(expected))


def test_find_amounts_reads_every_number_in_ocr_text():
    text = "1 REAL GANACHE 16,500\nTOTAL 45.500\nCASH Rp.50,000"
    assert {Decimal(16500), Decimal(45500), Decimal(50000), Decimal(1)} <= find_amounts(text)


def test_normalize_text_ignores_case_spacing_and_punctuation():
    assert normalize_text("JASMINE MT ( L )") == normalize_text("Jasmine Mt (L)")


def test_as_numbers_converts_every_amount():
    receipt = Receipt(
        items=[LineItem(name="Kopi", quantity="2x", unit_price="@9.000", price="18.000")],
        total="Rp 18.000",
    )
    values = as_numbers(receipt)
    assert values["total"] == 18000.0
    assert values["items"][0] == {
        "name": "Kopi", "code": None, "quantity": 2.0, "unit_price": 9000.0,
        "discount": None, "price": 18000.0, "sub_items": [],
    }  # fmt: skip
    assert values["subtotal"] is None
