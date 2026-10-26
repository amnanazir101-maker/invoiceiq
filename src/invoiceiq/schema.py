"""The receipt schema that every extractor fills and the API returns.

Values are kept exactly as printed ("Rp. 16.500", "2x", "@11,000"). Copying text is what vision
models do reliably; turning it into numbers is left to deterministic code (`amounts.py`), so
every number can be traced back to the characters it came from. The field descriptions are
also the instructions Gemini reads.
"""

from __future__ import annotations

from collections.abc import Iterator

from pydantic import BaseModel, Field, field_validator

_AS_PRINTED = "exactly as printed"


def _name(value: object) -> object:
    """A model that returns null for an unreadable name still yields a usable item."""
    return "" if value is None else value


class SubItem(BaseModel):
    """An add-on, option or set component printed under a line item."""

    name: str = Field(description=f"Name {_AS_PRINTED}.")
    _blank_name = field_validator("name", mode="before")(_name)
    quantity: str | None = Field(None, description=f"Quantity {_AS_PRINTED}.")
    unit_price: str | None = Field(None, description=f"Price of one unit {_AS_PRINTED}.")
    price: str | None = Field(None, description=f"Amount printed for this line, {_AS_PRINTED}.")


class LineItem(BaseModel):
    """One purchased product or service."""

    name: str = Field(description=f"Item name {_AS_PRINTED}.")
    _blank_name = field_validator("name", mode="before")(_name)
    code: str | None = Field(None, description="Item number, SKU or barcode printed with it.")
    quantity: str | None = Field(
        None, description="Quantity exactly as printed, with any attached 'x' ('2', '1X', '1 x')."
    )
    unit_price: str | None = Field(
        None, description="Price of one unit exactly as printed, with any '@' or currency prefix."
    )
    discount: str | None = Field(None, description="Discount printed for this item, with its sign.")
    price: str | None = Field(
        None, description=f"Amount printed for the whole line, {_AS_PRINTED}."
    )
    sub_items: list[SubItem] = Field(
        default_factory=list,
        description="Add-ons, options or set components printed under this item.",
    )


class Receipt(BaseModel):
    """Everything on a receipt, bill or invoice that bookkeeping needs."""

    merchant: str | None = Field(None, description="Business name as printed.")
    document_number: str | None = Field(
        None, description="Receipt, bill or invoice number as printed."
    )
    date: str | None = Field(None, description="Date (and time) as printed.")
    currency: str | None = Field(
        None, description="Currency symbol or code if printed, such as 'Rp' or 'USD'."
    )
    items: list[LineItem] = Field(
        default_factory=list, description="Every purchased item, top to bottom, once each."
    )
    subtotal: str | None = Field(None, description=f"Subtotal {_AS_PRINTED}.")
    discount: str | None = Field(
        None, description="Discount on the whole bill exactly as printed, with its sign."
    )
    service_charge: str | None = Field(None, description=f"Service charge {_AS_PRINTED}.")
    other_charges: str | None = Field(
        None, description="Any other surcharge (delivery, packaging) exactly as printed."
    )
    tax: str | None = Field(None, description="Tax amount (VAT, GST, PPN, PB1) exactly as printed.")
    total: str | None = Field(None, description="Final amount due exactly as printed.")
    cash: str | None = Field(None, description=f"Cash tendered, {_AS_PRINTED}.")
    change: str | None = Field(None, description=f"Change returned, {_AS_PRINTED}.")
    card: str | None = Field(None, description=f"Amount paid by card, {_AS_PRINTED}.")
    e_money: str | None = Field(
        None, description="Amount paid by e-wallet or QR payment exactly as printed."
    )
    item_count: str | None = Field(
        None, description="Total quantity of items printed on the receipt (e.g. 'Qty: 3')."
    )
    line_count: str | None = Field(
        None, description="Number of different items printed on the receipt."
    )

    def amount_fields(self) -> Iterator[tuple[str, str]]:
        """Every printed amount with its field path, such as ("items[2].price", "16,500")."""
        for index, item in enumerate(self.items):
            for name in ITEM_AMOUNTS:
                if value := getattr(item, name):
                    yield f"items[{index}].{name}", value
            for sub_index, sub in enumerate(item.sub_items):
                for name in ("unit_price", "price"):
                    if value := getattr(sub, name):
                        yield f"items[{index}].sub_items[{sub_index}].{name}", value
        for name in RECEIPT_AMOUNTS:
            if value := getattr(self, name):
                yield name, value


ITEM_AMOUNTS = ("unit_price", "discount", "price")
RECEIPT_AMOUNTS = (
    "subtotal", "discount", "service_charge", "other_charges", "tax",
    "total", "cash", "change", "card", "e_money",
)  # fmt: skip
