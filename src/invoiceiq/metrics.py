"""Scoring predictions against CORD labels.

Field-level F1 follows the protocol of the Donut paper (Kim et al., ECCV 2022), the standard way
CORD results are reported: both trees are flattened into (field path, value) pairs such as
("menu.price", "16,500"), and a predicted pair counts as correct only if the label has the same
pair, matched one-to-one, micro-averaged over the split. Values must match character for
character ("strict").

The normalized variant compares what a ledger would store: amounts as numbers ("Rp 16.500" =
"16500"), counts as numbers ("2x" = "2") and names without case, spacing or punctuation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from invoiceiq.amounts import canonical, normalize_text, parse_amount, parse_quantity

AMOUNT_KEYS = {
    "price", "unitprice", "discountprice", "itemsubtotal", "subtotal_price", "discount_price",
    "service_price", "othersvc_price", "tax_price", "total_price", "cashprice", "changeprice",
    "creditcardprice", "emoneyprice",
}  # fmt: skip
COUNT_KEYS = {"cnt", "menuqty_cnt", "menutype_cnt"}


def normalize_tree(data: object) -> object:
    """Drop empty values and wrap scalars in lists, as the Donut evaluator does."""
    if not data:
        return {}
    if isinstance(data, dict):
        tree = {}
        for key in sorted(data, key=lambda k: (len(k), k)):
            value = normalize_tree(data[key])
            if value:
                tree[key] = value if isinstance(value, list) else [value]
        return tree
    if isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            return [item for item in (normalize_tree(item) for item in data) if item]
        return [
            str(item).strip()
            for item in data
            if isinstance(item, str | int | float) and str(item).strip()
        ]
    return [str(data).strip()]


def flatten(data: object) -> list[tuple[str, str]]:
    """Every leaf of a CORD tree as a (dotted path, value) pair."""
    pairs: list[tuple[str, str]] = []

    def walk(value: object, key: str) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                walk(child, f"{key}.{child_key}" if key else child_key)
        elif isinstance(value, list):
            for child in value:
                walk(child, key)
        else:
            pairs.append((key, str(value)))

    walk(normalize_tree(data), "")
    return pairs


def normalize_value(key: str, value: str) -> str:
    leaf = key.rsplit(".", 1)[-1]
    if leaf in AMOUNT_KEYS and (amount := parse_amount(value)) is not None:
        return canonical(abs(amount) if "discount" in leaf else amount)
    if leaf in COUNT_KEYS and (count := parse_quantity(value)) is not None:
        return canonical(count)
    return normalize_text(value)


def pairs(tree: dict, normalized: bool = False) -> Counter:
    flat = flatten(tree)
    if normalized:
        flat = [(key, normalize_value(key, value)) for key, value in flat]
    return Counter(flat)


@dataclass
class Score:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: Score) -> None:
        self.tp, self.fp, self.fn = self.tp + other.tp, self.fp + other.fp, self.fn + other.fn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        """TP / (TP + (FP + FN) / 2), Donut's formula (the usual F1)."""
        return self.tp / (self.tp + (self.fp + self.fn) / 2) if self.tp + self.fp + self.fn else 0.0

    def as_dict(self) -> dict:
        return {
            "f1": round(self.f1, 4), "precision": round(self.precision, 4),
            "recall": round(self.recall, 4), "support": self.tp + self.fn,
        }  # fmt: skip


@dataclass
class FieldScores:
    overall: Score = field(default_factory=Score)
    by_field: dict[str, Score] = field(default_factory=dict)


def score_pair(pred: dict, gt: dict, normalized: bool = False) -> FieldScores:
    predicted, expected = pairs(pred, normalized), pairs(gt, normalized)
    scores = FieldScores()
    for key in {k for k, _ in predicted} | {k for k, _ in expected}:
        mine = Counter({p: n for p, n in predicted.items() if p[0] == key})
        theirs = Counter({p: n for p, n in expected.items() if p[0] == key})
        tp = sum((mine & theirs).values())
        score = Score(tp, sum(mine.values()) - tp, sum(theirs.values()) - tp)
        scores.by_field[key] = score
        scores.overall.add(score)
    return scores


def score_split(preds: list[dict], gts: list[dict], normalized: bool = False) -> FieldScores:
    total = FieldScores()
    for pred, gt in zip(preds, gts, strict=True):
        result = score_pair(pred, gt, normalized)
        total.overall.add(result.overall)
        for key, score in result.by_field.items():
            total.by_field.setdefault(key, Score()).add(score)
    return total


def total_correct(pred: dict, gt: dict) -> bool | None:
    """Whether the grand total is right (as a number); None when the label has no total."""
    expected = [v for k, v in flatten(gt) if k == "total.total_price"]
    if not expected:
        return None
    predicted = [v for k, v in flatten(pred) if k == "total.total_price"]
    if not predicted:
        return False
    return normalize_value("total_price", predicted[0]) == normalize_value(
        "total_price", expected[0]
    )


def amounts_exact(pred: dict, gt: dict) -> bool:
    """Every amount on the receipt is right: the bar for booking it without a human look."""

    def amounts(tree: dict) -> Counter:
        return Counter(
            p
            for p in pairs(tree, normalized=True).elements()
            if p[0].rsplit(".", 1)[-1] in AMOUNT_KEYS
        )

    return amounts(pred) == amounts(gt)
