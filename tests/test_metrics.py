import pytest

from invoiceiq.metrics import amounts_exact, flatten, score_pair, score_split, total_correct

GT = {
    "menu": [
        {"nm": "REAL GANACHE", "cnt": "1", "price": "16,500"},
        {"nm": "EGG TART", "cnt": "1", "price": "13,000"},
    ],
    "total": {"total_price": "29,500", "cashprice": "50,000", "changeprice": "20,500"},
}


def test_flatten_follows_the_donut_protocol():
    single_item = {"menu": {"nm": "Kopi", "price": "9.000"}, "total": {"total_price": ["9.000"]}}
    assert sorted(flatten(single_item)) == [
        ("menu.nm", "Kopi"), ("menu.price", "9.000"), ("total.total_price", "9.000"),
    ]  # fmt: skip
    assert flatten({"menu": [{"nm": "", "price": None}]}) == []  # empty values are dropped


def test_a_perfect_prediction_scores_one():
    scores = score_pair(GT, GT)
    assert scores.overall.f1 == 1.0
    assert scores.overall.tp == 9


def test_f1_counts_wrong_missing_and_extra_fields():
    pred = {
        "menu": [
            {"nm": "REAL GANACHE", "cnt": "1", "price": "16,500"},
            {"nm": "EGG TART", "cnt": "1", "price": "18,000"},  # misread price
        ],
        "total": {"total_price": "29,500", "cashprice": "50,000", "creditcardprice": "0"},
    }  # changeprice missing, creditcardprice extra
    scores = score_pair(pred, GT)
    assert (scores.overall.tp, scores.overall.fp, scores.overall.fn) == (7, 2, 2)
    assert scores.overall.f1 == pytest.approx(7 / (7 + 2))
    assert scores.by_field["menu.price"].f1 == pytest.approx(0.5)


def test_normalized_scoring_ignores_formatting():
    pred = {
        "menu": [
            {"nm": "Real Ganache", "cnt": "1x", "price": "16500"},
            {"nm": "EGG TART", "cnt": "1", "price": "Rp 13.000"},
        ],
        "total": {"total_price": "29.500", "cashprice": "50,000", "changeprice": "20,500.00"},
    }
    assert score_pair(pred, GT).overall.f1 < 0.7
    assert score_pair(pred, GT, normalized=True).overall.f1 == 1.0


def test_score_split_micro_averages():
    perfect, empty = score_pair(GT, GT), score_pair({}, GT)
    total = score_split([GT, {}], [GT, GT])
    assert total.overall.tp == perfect.overall.tp
    assert total.overall.fn == empty.overall.fn == 9
    assert total.overall.f1 == pytest.approx(9 / (9 + 9 / 2))


def test_total_correct_compares_numbers():
    assert total_correct({"total": {"total_price": "Rp 29.500"}}, GT) is True
    assert total_correct({"total": {"total_price": "29,000"}}, GT) is False
    assert total_correct({}, GT) is False
    assert total_correct({}, {"menu": {"nm": "x"}}) is None


def test_amounts_exact_needs_every_amount():
    assert amounts_exact(GT, GT)
    wrong_change = {**GT, "total": {**GT["total"], "changeprice": "25,500"}}
    assert not amounts_exact(wrong_change, GT)
    renamed = {**GT, "menu": [{**item, "nm": item["nm"].lower()} for item in GT["menu"]]}
    assert amounts_exact(renamed, GT)  # names are not amounts
