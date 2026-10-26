from invoiceiq.cord import from_cord, to_cord
from invoiceiq.metrics import score_pair
from invoiceiq.schema import LineItem, Receipt, SubItem

LABEL = {
    "menu": {
        "nm": "M-Caramel Black Tea",
        "unitprice": "@28,000",
        "cnt": "1X",
        "price": "28,000",
        "sub": [{"nm": "70%"}, {"nm": "Less Ice"}],
    },
    "sub_total": {"subtotal_price": "28,000", "tax_price": "0"},
    "total": {"total_price": "28,000", "cashprice": "28,000", "changeprice": "0"},
}


def test_from_cord_reads_a_single_item_label():
    receipt = from_cord(LABEL)
    assert receipt.items == [
        LineItem(
            name="M-Caramel Black Tea", quantity="1X", unit_price="@28,000", price="28,000",
            sub_items=[SubItem(name="70%"), SubItem(name="Less Ice")],
        )
    ]  # fmt: skip
    assert (receipt.subtotal, receipt.tax, receipt.total) == ("28,000", "0", "28,000")
    assert (receipt.cash, receipt.change) == ("28,000", "0")


def test_round_trip_scores_perfectly():
    assert score_pair(to_cord(from_cord(LABEL)), LABEL).overall.f1 == 1.0


def test_to_cord_maps_every_field():
    receipt = Receipt(
        merchant="Toko Kopi",  # not part of CORD's labels
        items=[
            LineItem(name="Kopi", code="901016", quantity="2", discount="-2,000", price="18,000")
        ],
        subtotal="18,000", discount="-2,000", service_charge="900", other_charges="500",
        tax="1,600", total="19,000", cash="20,000", change="1,000", card="0", e_money="0",
        item_count="2", line_count="1",
    )  # fmt: skip
    assert to_cord(receipt) == {
        "menu": [
            {"nm": "Kopi", "num": "901016", "cnt": "2", "discountprice": "-2,000",
             "price": "18,000"},
        ],
        "sub_total": {
            "subtotal_price": "18,000", "discount_price": "-2,000", "service_price": "900",
            "othersvc_price": "500", "tax_price": "1,600",
        },
        "total": {
            "total_price": "19,000", "cashprice": "20,000", "changeprice": "1,000",
            "creditcardprice": "0", "emoneyprice": "0", "menuqty_cnt": "2", "menutype_cnt": "1",
        },
    }  # fmt: skip


def test_lists_and_multi_line_names_are_joined():
    receipt = from_cord(
        {
            "menu": [{"nm": ["NASI", "GORENG"], "price": "25.000"}],
            "total": [{"total_price": "25.000"}],
        }
    )
    assert receipt.items[0].name == "NASI GORENG"
    assert receipt.total == "25.000"
