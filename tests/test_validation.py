from invoiceiq.schema import LineItem, Receipt, SubItem
from invoiceiq.validation import validate


def receipt(**overrides) -> Receipt:
    fields = {
        "items": [
            LineItem(name="Kopi Susu", quantity="2", unit_price="@9,000", price="18,000"),
            LineItem(name="Roti Bakar", quantity="1", price="30,000"),
        ],
        "subtotal": "48,000",
        "tax": "4,800",
        "total": "52,800",
        "cash": "100,000",
        "change": "47,200",
    }
    fields.update(overrides)
    return Receipt(**fields)


def statuses(review) -> dict[str, str]:
    return {check.name: check.status for check in review.checks}


def test_a_consistent_receipt_is_approved():
    review = validate(receipt())
    assert review.status == "approved"
    assert statuses(review) == {
        "Total found": "pass",
        "Items add up": "pass",
        "Totals add up": "pass",
        "Change is right": "pass",
        "Quantity times price": "pass",
        "Item count": "skip",
    }
    assert review.flagged == []


def test_a_misread_total_is_sent_to_review_with_the_fields_to_check():
    review = validate(receipt(total="58,800"))
    assert review.status == "needs_review"
    failed = {check.name for check in review.failed}
    assert failed == {"Totals add up", "Change is right"}
    assert {"total", "subtotal", "tax", "cash", "change"} <= set(review.flagged)


def test_a_missed_item_breaks_the_item_sum():
    review = validate(receipt(items=receipt().items[:1]))
    assert review.status == "needs_review"
    assert [check.name for check in review.failed] == ["Items add up"]


def test_tax_included_in_the_prices_is_accepted():
    # Prices include tax: the items add up to the total, and the subtotal is the net amount.
    review = validate(
        Receipt(
            items=[LineItem(name="S-Ovaltine", price="20,000")],
            subtotal="18,181",
            tax="1,818",
            total="20,000",
        )
    )
    assert review.status == "approved"


def test_discount_and_rounding_within_tolerance():
    review = validate(
        Receipt(
            items=[
                LineItem(name="Nasi Campur", price="75,000"),
                LineItem(name="Es Teh", price="16,045"),
            ],
            subtotal="91,045",
            discount="-10,000",
            total="81,000",  # a Rp 45 rounding line
        )
    )
    assert review.status == "approved"


def test_add_on_prices_count_towards_the_items():
    review = validate(
        Receipt(
            items=[
                LineItem(
                    name="Milk Tea", price="25,000", sub_items=[SubItem(name="Boba", price="5,000")]
                )
            ],
            total="30,000",
        )
    )
    assert statuses(review)["Items add up"] == "pass"


def test_wrong_line_math_flags_that_line():
    review = validate(
        receipt(items=[LineItem(name="Kopi", quantity="2", unit_price="9,000", price="19,000")])
    )
    line = next(check for check in review.checks if check.name == "Quantity times price")
    assert line.status == "fail"
    assert "items[0].price" in line.fields


def test_printed_item_count():
    good = validate(receipt(item_count="3"))  # 2 + 1
    bad = validate(receipt(item_count="5"))
    assert statuses(good)["Item count"] == "pass"
    assert statuses(bad)["Item count"] == "fail"


def test_no_total_fails():
    review = validate(Receipt(items=[LineItem(name="Kopi", price="9,000")]))
    assert review.status == "needs_review"
    assert "total" in review.flagged


def test_amounts_missing_from_the_ocr_text_are_flagged():
    ocr_text = (
        "KOPI SUSU 2 @9,000 18,000\nROTI BAKAR 1 30,000\nSUBTOTAL 48,000\nPB1 4,800\nTOTAL 52,800"
    )
    ok = validate(receipt(cash=None, change=None), ocr_text=ocr_text)
    invented = validate(
        receipt(cash=None, change=None, tax="4,300", total="52,300"), ocr_text=ocr_text
    )
    assert ok.status == "approved"
    printed = next(check for check in invented.checks if check.name == "Amounts are printed")
    assert printed.status == "fail"
    assert set(printed.fields) == {"tax", "total"}


def test_only_a_total_is_approved_when_it_is_printed():
    review = validate(Receipt(total="23.000"), ocr_text="Kopi Susu Kolonel 23.000\nTotal 23.000")
    assert review.status == "approved"
