"""Business-rule checks that decide whether a receipt can be booked without a human look.

A vision model can misread one digit or skip a line and still return perfectly valid JSON.
These checks recompute what the document itself promises: the items add up to the subtotal,
the subtotal plus tax and service is the total, cash minus the total is the change. When OCR
text is available they also confirm that every amount is actually printed on the page, which
catches values the model invented. A receipt with any failed check goes to a reviewer, with the
fields to look at.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, Field, computed_field

from invoiceiq.amounts import canonical, find_amounts, parse_amount, parse_quantity, readable
from invoiceiq.schema import Receipt

# 0.5% of the bill absorbs rounding lines (Rp 45 on Rp 91,045). A misread digit small enough to
# hide inside it is left to the printed-amounts check.
DEFAULT_TOLERANCE = 0.005


class Check(BaseModel):
    name: str
    status: Literal["pass", "fail", "skip"]  # skip: the fields it needs are not on the receipt
    detail: str
    fields: list[str] = Field(default_factory=list)  # what a reviewer should look at on a fail


class Review(BaseModel):
    status: Literal["approved", "needs_review"]
    checks: list[Check]

    @computed_field
    @property
    def flagged(self) -> list[str]:
        """Fields involved in a failed check."""
        return sorted({field for check in self.failed for field in check.fields})

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if check.status == "fail"]


def validate(
    receipt: Receipt, ocr_text: str | None = None, tolerance: float = DEFAULT_TOLERANCE
) -> Review:
    """Run every check; `ocr_text` (the page as read by OCR) enables the printed-amounts check."""
    tol = Decimal(str(tolerance))
    checks = [
        _total_found(receipt),
        _items_add_up(receipt, tol),
        _totals_add_up(receipt, tol),
        _change(receipt, tol),
        _line_math(receipt, tol),
        _counts(receipt),
    ]
    if ocr_text is not None:
        checks.append(_printed(receipt, ocr_text))
    failed = any(check.status == "fail" for check in checks)
    return Review(status="needs_review" if failed else "approved", checks=checks)


def _close(a: Decimal, b: Decimal, tol: Decimal, scale: Decimal | None = None) -> bool:
    """Equal within `tol` of `scale` (by default the larger of the two amounts)."""
    reference = abs(scale) if scale is not None else max(abs(a), abs(b))
    return abs(a - b) <= max(Decimal("0.01"), reference * tol)


def _amount(receipt: Receipt, name: str) -> Decimal | None:
    return parse_amount(getattr(receipt, name))


def _present(receipt: Receipt, names: Iterable[str]) -> list[str]:
    return [name for name in names if getattr(receipt, name)]


def bill_totals(base: Decimal, receipt: Receipt) -> Iterator[Decimal]:
    """Every total the receipt could print after `base`: any of tax, service and other charges
    added (taxes are sometimes already included), with and without the bill discount."""
    charges = [
        value
        for name in ("tax", "service_charge", "other_charges")
        if (value := _amount(receipt, name))
    ]
    discount = abs(_amount(receipt, "discount") or 0)
    for size in range(len(charges) + 1):
        for combo in itertools.combinations(charges, size):
            amount = base + sum(combo, Decimal(0))
            yield amount
            if discount:
                yield amount - discount


def _total_found(receipt: Receipt) -> Check:
    name = "Total found"
    total = _amount(receipt, "total")
    if total is None:
        return Check(name=name, status="fail", detail="No total was read.", fields=["total"])
    if total <= 0:
        return Check(
            name=name, status="fail", detail=f"The total is {receipt.total}.", fields=["total"]
        )
    return Check(name=name, status="pass", detail=f"Total {receipt.total}.")


def _items_add_up(receipt: Receipt, tol: Decimal) -> Check:
    name = "Items add up"
    prices = [parse_amount(item.price) for item in receipt.items]
    if not any(price is not None for price in prices):
        return Check(name=name, status="skip", detail="No item prices to add up.")
    base = sum((price for price in prices if price is not None), Decimal(0))
    discounts = sum((abs(parse_amount(i.discount) or 0) for i in receipt.items), Decimal(0))
    add_ons = sum(
        (parse_amount(s.price) or 0 for i in receipt.items for s in i.sub_items), Decimal(0)
    )
    sums = {base, base - discounts, base + add_ons, base + add_ons - discounts}
    subtotal, total = _amount(receipt, "subtotal"), _amount(receipt, "total")
    if subtotal is None and total is None:
        return Check(name=name, status="skip", detail="No subtotal or total to compare with.")
    # Item prices add up to the subtotal, or (with tax-inclusive prices) straight to the total.
    if subtotal is not None and any(_close(s, subtotal, tol) for s in sums):
        return Check(name=name, status="pass", detail=f"Items add up to {receipt.subtotal}.")
    if total is not None and any(
        _close(t, total, tol) for s in sums for t in bill_totals(s, receipt)
    ):
        return Check(name=name, status="pass", detail=f"Items add up to {receipt.total}.")
    target = "subtotal" if subtotal is not None else "total"
    printed = getattr(receipt, target)
    return Check(
        name=name,
        status="fail",
        detail=f"Items add up to {readable(base)}, but the {target} is {printed}.",
        fields=[target, "items"],
    )


def _totals_add_up(receipt: Receipt, tol: Decimal) -> Check:
    name = "Totals add up"
    subtotal, total = _amount(receipt, "subtotal"), _amount(receipt, "total")
    if subtotal is None or total is None:
        return Check(name=name, status="skip", detail="Needs both a subtotal and a total.")
    if any(_close(amount, total, tol) for amount in bill_totals(subtotal, receipt)):
        return Check(
            name=name, status="pass", detail="Subtotal, charges and discount match the total."
        )
    fields = _present(
        receipt, ("subtotal", "discount", "service_charge", "other_charges", "tax", "total")
    )
    return Check(
        name=name,
        status="fail",
        detail=f"No combination of subtotal {receipt.subtotal} and the printed charges gives "
        f"the total {receipt.total}.",
        fields=fields,
    )


def _change(receipt: Receipt, tol: Decimal) -> Check:
    name = "Change is right"
    cash, change, total = (_amount(receipt, n) for n in ("cash", "change", "total"))
    if cash is None or change is None or total is None:
        return Check(name=name, status="skip", detail="Needs cash, change and a total.")
    other = (_amount(receipt, "card") or 0) + (_amount(receipt, "e_money") or 0)
    change = abs(change)  # some tills print the change as a negative amount
    # Rounding of a cash payment is a share of the bill, not of the (often small) change.
    if any(_close(paid - total, change, tol, scale=total) for paid in (cash, cash + other)):
        return Check(name=name, status="pass", detail="Cash minus the total is the change.")
    return Check(
        name=name,
        status="fail",
        detail=f"Cash {receipt.cash} minus total {receipt.total} is {readable(cash - total)}, "
        f"not {receipt.change}.",
        fields=["cash", "change", "total"],
    )


def _line_math(receipt: Receipt, tol: Decimal) -> Check:
    name = "Quantity times price"
    checked, wrong = 0, []
    for index, item in enumerate(receipt.items):
        quantity = parse_quantity(item.quantity)
        unit, price = parse_amount(item.unit_price), parse_amount(item.price)
        if quantity is None or unit is None or price is None:
            continue
        checked += 1
        discount = abs(parse_amount(item.discount) or 0)
        if not any(
            _close(amount, price, tol) for amount in (quantity * unit, quantity * unit - discount)
        ):
            wrong.append(index)
    if not checked:
        return Check(name=name, status="skip", detail="No item prints a quantity and a unit price.")
    if not wrong:
        return Check(name=name, status="pass", detail=f"{checked} line(s) multiply out.")
    names = ", ".join(receipt.items[i].name for i in wrong)
    return Check(
        name=name,
        status="fail",
        detail=f"Quantity x unit price is not the line amount for: {names}.",
        fields=[
            f"items[{i}].{part}" for i in wrong for part in ("quantity", "unit_price", "price")
        ],
    )


def _counts(receipt: Receipt) -> Check:
    name = "Item count"
    item_count, line_count = parse_quantity(receipt.item_count), parse_quantity(receipt.line_count)
    if item_count is None and line_count is None:
        return Check(name=name, status="skip", detail="No item count is printed.")
    quantities = [parse_quantity(item.quantity) or Decimal(1) for item in receipt.items]
    with_add_ons = sum(quantities, Decimal(0)) + sum(
        (parse_quantity(s.quantity) or Decimal(1) for i in receipt.items for s in i.sub_items),
        Decimal(0),
    )
    wrong = []
    if item_count is not None and item_count not in (
        sum(quantities, Decimal(0)),
        len(receipt.items),
        with_add_ons,
    ):
        wrong.append("item_count")
    if line_count is not None and line_count != len(receipt.items):
        wrong.append("line_count")
    if not wrong:
        return Check(name=name, status="pass", detail="The printed item count matches the items.")
    return Check(
        name=name,
        status="fail",
        detail=f"{len(receipt.items)} item(s) were read, but the receipt prints a different count.",
        fields=[*wrong, "items"],
    )


def _printed(receipt: Receipt, ocr_text: str) -> Check:
    name = "Amounts are printed"
    printed = {canonical(value) for value in find_amounts(ocr_text)}
    missing = []
    for field, text in receipt.amount_fields():
        value = parse_amount(text)
        if value and canonical(abs(value)) not in printed:
            missing.append(field)
    if not missing:
        return Check(name=name, status="pass", detail="Every amount appears in the OCR text.")
    return Check(
        name=name,
        status="fail",
        detail=f"Not found in the OCR text: {', '.join(missing)}.",
        fields=missing,
    )
