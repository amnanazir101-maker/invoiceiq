"""The classic pipeline, as a baseline: OCR, then keyword and layout rules.

This is how receipt capture was built before vision-language models: read the text, find
lines such as "TOTAL" or "Kembali" (change) by keyword, take the amount at the end of the line,
and treat the lines above the first summary line as items. It needs no API and no training,
and it is brittle in the usual ways: skewed photos merge lines, wrapped names split items, and
unseen wording is missed. The keywords cover English and Indonesian receipts.
"""

from __future__ import annotations

import re
import time

from invoiceiq.documents import Document
from invoiceiq.extractors.base import Extraction
from invoiceiq.ocr import OcrEngine
from invoiceiq.schema import LineItem, Receipt, SubItem

# Summary lines, checked in order: "Total item" before "Total", "Subtotal" before "Total".
SUMMARY_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    (field, re.compile(pattern, re.IGNORECASE))
    for field, pattern in [
        ("item_count", r"total\s*(item|qty)|\bqty\b|\bitems?\b|jml\s*item"),
        ("subtotal", r"sub\s*-?\s*t(o)?t(a)?l"),
        ("discount", r"\bdisc|diskon|potongan"),
        ("tax", r"\btax\b|pajak|\bppn\b|\bpb\s*1\b|\bvat\b|\bgst\b"),
        ("service_charge", r"service|\bsvc\b|\bserv\b|layanan"),
        ("total", r"\btotal\b|\bttl\b|amount\s*due|tagihan"),
        ("change", r"change|kembali"),
        ("cash", r"\bcash\b|tunai|bayar"),
        ("card", r"\bcard\b|credit|debit|\bedc\b|visa|master|\bbca\b|mandiri|\bbni\b|\bbri\b"),
        ("e_money", r"gopay|\bovo\b|\bdana\b|shopee\s*pay|qris|e-?money|linkaja"),
    ]
]

# Money as receipts print it: 16,500 / 23.000 / @28,000 / Rp.15.000 / 48982.68 / 100000.
# Dates, times and percentages are not money.
MONEY = re.compile(
    r"""(?<![\d/:.,])
        (?:@\s?)?(?:Rp\.?\s?)?-?
        (?:\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?   # grouped thousands
          |\d+[.,]\d{2}                            # two decimals
          |[1-9]\d{3,8})                           # plain digits (not phone numbers)
        (?![\d/:%]|[.,]\d)""",
    re.VERBOSE | re.IGNORECASE,
)
QUANTITY = re.compile(r"^\s*(\d{1,3}(?:[.,]\d{1,2})?\s?[xX]?|[xX]\s?\d{1,3})(?=\s)")
LETTERS = re.compile(r"[A-Za-z]{2,}")
ZERO = re.compile(r"(?<![\d.,])0(?:[.,]0{1,2})?(?![\d.,%])")  # "Change 0" or "0.00"
COUNT = re.compile(r"(?<![\d.,])\d{1,3}(?:[.,]\d{1,2})?(?![\d.,%])")
TRAILING_NUMBER = re.compile(r"\s(\d{1,3}(?:[.,]\d{3})*)\s*$")


class OcrRulesExtractor:
    name = "ocr-rules"
    uses_ocr = True

    def __init__(self, ocr: OcrEngine | None = None) -> None:
        self.ocr = ocr or OcrEngine()

    def extract(self, document: Document) -> Extraction:
        started = time.perf_counter()
        lines = []
        for page in document.pages:
            lines += [" ".join(box.text for box in line) for line in self.ocr.read(page).lines()]
        return Extraction(
            receipt=parse_lines(lines),
            model="RapidOCR PP-OCR + rules",
            seconds=round(time.perf_counter() - started, 2),
            ocr_text="\n".join(lines),
        )


def summary_fields(line: str) -> dict[str, str | None]:
    """The summary fields a line names, with their values ("TOTAL 2 ITEMS 45,500" -> both)."""
    found: dict[str, str | None] = {}
    money = [m.group().strip() for m in MONEY.finditer(line)]
    for field, pattern in SUMMARY_KEYWORDS:
        if not pattern.search(line):
            continue
        if field == "item_count":
            counts = COUNT.findall(line)
            if counts:
                found[field] = counts[-1]
        elif not any(f != "item_count" for f in found):  # one amount field per line
            zero = ZERO.search(line)
            found[field] = money[0] if money else zero.group() if zero else None
    return found


def parse_lines(lines: list[str]) -> Receipt:
    """Turn OCR lines (top to bottom, each read left to right) into a Receipt."""
    fields: dict[str, str] = {}
    items: list[LineItem] = []
    in_summary = False
    pending_field: str | None = None  # a summary label whose amount wrapped onto the next line
    pending_name: str | None = None  # an item name whose quantity and price follow on the next line

    def flush_name() -> None:  # a text-only line not followed by a price: an add-on
        nonlocal pending_name
        if pending_name and items:
            items[-1].sub_items.append(_sub_item(pending_name))
        pending_name = None

    for line in lines:
        money = [m.group().strip() for m in MONEY.finditer(line)]
        has_letters = LETTERS.search(line) is not None
        # A "Qty  Item  Price" header names a count but holds none: it is not a summary line.
        found = summary_fields(line)
        if found:
            flush_name()
            in_summary = True
            pending_field = None
            for field, value in found.items():
                if value:
                    fields.setdefault(field, value)
                else:
                    pending_field = field
            continue
        if pending_field and (money or ZERO.search(line)) and not has_letters:
            fields.setdefault(pending_field, money[0] if money else ZERO.search(line).group())
            pending_field = None
            continue
        pending_field = None
        if in_summary:
            continue
        if money and not has_letters and pending_name:  # "Nasi Goreng" then "2 x 25.000 50.000"
            items.append(_item(f"{_quantity_prefix(line)}{pending_name}", money))
            pending_name = None
        elif money and has_letters:
            flush_name()
            items.append(_item(line, money))
        elif money and items and items[-1].price is None:  # a price wrapped onto its own line
            items[-1].price = money[-1]
        elif has_letters:
            flush_name()
            pending_name = line.strip()
    flush_name()
    return Receipt(items=items, **fields)


def _quantity_prefix(line: str) -> str:
    match = QUANTITY.match(line)
    return f"{match.group(1).strip()} " if match else ""


def _item(line: str, money: list[str]) -> LineItem:
    name = MONEY.sub(" ", line)
    quantity = None
    if match := QUANTITY.match(name):
        quantity = match.group(1).strip()
        name = name[match.end() :]
    return LineItem(
        name=" ".join(name.split()),
        quantity=quantity,
        unit_price=money[-2] if len(money) > 1 else None,
        price=money[-1],
    )


def _sub_item(line: str) -> SubItem:
    """An add-on line such as "1 ROASTED ALMOND 0" or "- Less Ice"."""
    quantity = price = None
    if match := QUANTITY.match(line):
        quantity, line = match.group(1).strip(), line[match.end() :]
    if match := TRAILING_NUMBER.search(line):
        price, line = match.group(1), line[: match.start()]
    return SubItem(name=" ".join(line.split()), quantity=quantity, price=price)
