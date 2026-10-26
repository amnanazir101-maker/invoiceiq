"""Turning printed text into numbers: '16.500', '16,500.00', 'Rp. 16.500,-', '@11000', '(5,000)'.

Receipts mix conventions: Indonesian receipts group thousands with dots ("16.500"), US ones with
commas ("16,500.00"), some with spaces ("2 600"). A single separator followed by exactly three
digits is read as a thousands separator; one or two digits make it a decimal mark.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from decimal import Decimal, InvalidOperation

from invoiceiq.schema import RECEIPT_AMOUNTS, Receipt

# Digits with "." or "," inside, plus space-separated groups of exactly three digits ("2 600").
_NUMBER = re.compile(r"\d(?:[\d.,]*\d)?(?: \d{3}(?![\d.,]))*")
# OCR text: the same without spaces, which there separate neighbouring numbers.
_OCR_NUMBER = re.compile(r"\d(?:[\d.,]*\d)?")
_QUANTITY = re.compile(r"\d+(?:[.,]\d+)?")
_MINUS = ("-", "−")


def to_decimal(token: str) -> Decimal | None:
    """Read one number token, working out which separator is the decimal mark."""
    token = token.replace(" ", "")
    dot, comma = token.rfind("."), token.rfind(",")
    if dot >= 0 and comma >= 0:  # both: the last one is the decimal mark (16,500.00 / 16.500,00)
        mark, group = (".", ",") if dot > comma else (",", ".")
        token = token.replace(group, "").replace(mark, ".")
    elif dot >= 0 or comma >= 0:
        separator = "." if dot >= 0 else ","
        groups = token.split(separator)
        if len(groups) > 2 or len(groups[-1]) == 3:  # 16.500, 1,234,567
            token = token.replace(separator, "")
        else:  # 4.5, 12,50
            token = token.replace(separator, ".")
    try:
        return Decimal(token)
    except InvalidOperation:
        return None


def parse_amount(text: str | None) -> Decimal | None:
    """The amount in a printed value; the last number wins ('Rp. 1.500' -> 1500).

    A minus sign (before the number or right after it) or surrounding parentheses make it
    negative; the Indonesian ',-' suffix ('16.500,-') does not. Percentages are skipped, so
    'Disc 10% 5.000' reads as 5000.
    """
    if not text:
        return None
    matches = [m for m in _NUMBER.finditer(text) if not text[m.end() :].lstrip().startswith("%")]
    if not matches:
        return None
    match = matches[-1]
    value = to_decimal(match.group())
    if value is None:
        return None
    before, after = text[: match.start()], text[match.end() :]
    negative = (
        any(sign in before for sign in _MINUS)
        or after.startswith(_MINUS)
        or ("(" in before and ")" in after)
    )
    return -value if negative and value else value


def parse_quantity(text: str | None) -> Decimal | None:
    """The first number in a quantity ('2', '1X', 'x2', '1.00xITEMs' -> 2, 1, 2, 1)."""
    if not text:
        return None
    match = _QUANTITY.search(text)
    return Decimal(match.group().replace(",", ".")) if match else None


def find_amounts(text: str) -> set[Decimal]:
    """Every number printed in a block of OCR text, as absolute values.

    Numbers are read both with and without space-grouped thousands, so "18 000" yields 18000
    as well as 18 and 0: a reading that only adds candidates never hides a printed amount.
    """
    found = set()
    for pattern in (_OCR_NUMBER, _NUMBER):
        for match in pattern.finditer(text):
            value = to_decimal(match.group())
            if value is not None:
                found.add(abs(value))
    return found


def canonical(value: Decimal) -> str:
    """A comparable string: 16500.00 -> '16500', 4.50 -> '4.5'."""
    text = format(value.normalize(), "f")
    return "0" if text in ("0", "-0") else text


def readable(value: Decimal) -> str:
    """For messages: 132000 -> '132,000', 4.50 -> '4.5'."""
    text = format(value.normalize(), ",f")
    return "0" if text in ("0", "-0") else text


def normalize_text(text: str) -> str:
    """Case, spacing and punctuation removed, for comparing names ('Ice Tea (L)' ~ 'ICE TEA L')."""
    return re.sub(r"[\W_]+", "", text.casefold())


def to_number(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def as_numbers(receipt: Receipt) -> dict:
    """The receipt with every amount and quantity as a number, ready for a ledger or a sheet."""
    return {
        "merchant": receipt.merchant,
        "document_number": receipt.document_number,
        "date": receipt.date,
        "currency": receipt.currency,
        "items": [
            {
                "name": item.name,
                "code": item.code,
                "quantity": to_number(parse_quantity(item.quantity)),
                "unit_price": to_number(parse_amount(item.unit_price)),
                "discount": to_number(parse_amount(item.discount)),
                "price": to_number(parse_amount(item.price)),
                "sub_items": [
                    {
                        "name": sub.name,
                        "quantity": to_number(parse_quantity(sub.quantity)),
                        "unit_price": to_number(parse_amount(sub.unit_price)),
                        "price": to_number(parse_amount(sub.price)),
                    }
                    for sub in item.sub_items
                ],
            }
            for item in receipt.items
        ],
        **{name: to_number(parse_amount(getattr(receipt, name))) for name in RECEIPT_AMOUNTS},
        "item_count": to_number(parse_quantity(receipt.item_count)),
        "line_count": to_number(parse_quantity(receipt.line_count)),
    }


def iter_amounts(receipt: Receipt) -> Iterator[tuple[str, Decimal]]:
    """Every parseable amount on the receipt with its field path."""
    for field, text in receipt.amount_fields():
        value = parse_amount(text)
        if value is not None:
            yield field, value
