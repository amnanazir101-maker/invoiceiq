"""The CORD v2 benchmark (Park et al., 2019): 1,000 Indonesian receipts with labelled fields.

Hosted on Hugging Face as `naver-clova-ix/cord-v2` (CC BY 4.0): 800 train, 100 validation and
100 test receipts. Each label (`gt_parse`) is a JSON tree with `menu`, `sub_total` and `total`
groups whose values are copied verbatim from the receipt. This module downloads one split and
maps between that tree and the `Receipt` schema.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from invoiceiq.schema import LineItem, Receipt, SubItem

REPO = "naver-clova-ix/cord-v2"
SPLITS = ("train", "validation", "test")

# Receipt field <-> CORD key, per group.
ITEM_KEYS = {
    "name": "nm", "code": "num", "quantity": "cnt", "unit_price": "unitprice",
    "discount": "discountprice", "price": "price",
}  # fmt: skip
SUB_ITEM_KEYS = {"name": "nm", "quantity": "cnt", "unit_price": "unitprice", "price": "price"}
SUB_TOTAL_KEYS = {
    "subtotal": "subtotal_price", "discount": "discount_price", "service_charge": "service_price",
    "other_charges": "othersvc_price", "tax": "tax_price",
}  # fmt: skip
TOTAL_KEYS = {
    "total": "total_price", "cash": "cashprice", "change": "changeprice",
    "card": "creditcardprice", "e_money": "emoneyprice", "item_count": "menuqty_cnt",
    "line_count": "menutype_cnt",
}  # fmt: skip


@dataclass
class Sample:
    id: str  # "test-007"
    image_bytes: bytes
    gt: dict  # the gt_parse tree

    @property
    def image(self) -> Image.Image:
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")


def load_split(split: str = "test", cache_dir: str | Path | None = None) -> list[Sample]:
    """Download (once) and read one split. Only that split's parquet file is fetched."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}")
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        name
        for name in HfApi().list_repo_files(REPO, repo_type="dataset")
        if name.startswith(f"data/{split}-") and name.endswith(".parquet")
    )
    samples: list[Sample] = []
    for name in files:
        path = hf_hub_download(REPO, name, repo_type="dataset", cache_dir=cache_dir)
        for row in pq.read_table(path).to_pylist():
            label = json.loads(row["ground_truth"])
            samples.append(
                Sample(f"{split}-{len(samples):03d}", row["image"]["bytes"], label["gt_parse"])
            )
    return samples


def _as_list(value: object) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _text(value: object) -> str | None:
    """A CORD value as one string; a field split over several lines is joined with spaces."""
    parts = [str(part).strip() for part in _as_list(value) if str(part).strip()]
    return " ".join(parts) if parts else None


def from_cord(tree: dict) -> Receipt:
    """A CORD label (or a Donut prediction) as a Receipt."""
    items = []
    for entry in _as_list(tree.get("menu")):
        if not isinstance(entry, dict):
            continue
        subs = [
            SubItem(
                name=_text(sub.get("nm")) or "",
                **{
                    field: _text(sub.get(key))
                    for field, key in SUB_ITEM_KEYS.items()
                    if field != "name"
                },
            )
            for sub in _as_list(entry.get("sub"))
            if isinstance(sub, dict)
        ]
        items.append(
            LineItem(
                name=_text(entry.get("nm")) or "",
                sub_items=subs,
                **{
                    field: _text(entry.get(key))
                    for field, key in ITEM_KEYS.items()
                    if field != "name"
                },
            )
        )
    groups = {"sub_total": SUB_TOTAL_KEYS, "total": TOTAL_KEYS}
    fields = {}
    for group, keys in groups.items():
        values = tree.get(group)
        values = values[0] if isinstance(values, list) and values else values
        if isinstance(values, dict):
            fields.update({field: _text(values.get(key)) for field, key in keys.items()})
    return Receipt(items=items, **fields)


def to_cord(receipt: Receipt) -> dict:
    """A Receipt as a CORD tree, so it can be scored against the labels."""

    def pick(obj: object, keys: dict[str, str]) -> dict:
        return {key: value for field, key in keys.items() if (value := getattr(obj, field))}

    menu = []
    for item in receipt.items:
        entry = pick(item, ITEM_KEYS)
        subs = [pick(sub, SUB_ITEM_KEYS) for sub in item.sub_items]
        if subs := [sub for sub in subs if sub]:
            entry["sub"] = subs
        if entry:
            menu.append(entry)
    tree: dict = {}
    if menu:
        tree["menu"] = menu
    if sub_total := pick(receipt, SUB_TOTAL_KEYS):
        tree["sub_total"] = sub_total
    if total := pick(receipt, TOTAL_KEYS):
        tree["total"] = total
    return tree
