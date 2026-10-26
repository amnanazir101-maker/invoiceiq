"""InvoiceIQ web app. Run with: streamlit run app.py

Upload receipts or invoices (photos, scans or PDFs), see what was read and which checks pass,
correct anything that is flagged, and export everything to Excel or JSON.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from invoiceiq import bench
from invoiceiq.charts import TOKENS, benchmark_chart, benchmark_frame, style
from invoiceiq.config import Settings
from invoiceiq.documents import SUFFIXES, Document, DocumentError
from invoiceiq.export import to_excel, to_json
from invoiceiq.extractors import ExtractionError
from invoiceiq.pipeline import Pipeline, Result
from invoiceiq.schema import LineItem, Receipt

st.set_page_config(page_title="InvoiceIQ", page_icon="🧾", layout="wide")
SETTINGS = Settings.from_env()
EXAMPLES = Path(__file__).parent / "examples"
READERS = {
    "Gemini (free tier)": "gemini",
    "Donut, fine-tuned on CORD (local)": "donut",
    "OCR + rules (local)": "ocr-rules",
}
HEADER = {
    "merchant": "Merchant",
    "document_number": "Number",
    "date": "Date",
    "currency": "Currency",
}
AMOUNTS = {
    "subtotal": "Subtotal", "discount": "Discount", "service_charge": "Service charge",
    "other_charges": "Other charges", "tax": "Tax", "total": "Total", "cash": "Cash",
    "change": "Change", "card": "Card", "e_money": "E-money",
}  # fmt: skip
ITEM_COLUMNS = {"name": "Item", "quantity": "Qty", "unit_price": "Unit price", "price": "Amount"}
STATE = st.session_state
for name in ("results", "files", "receipts", "reviews", "approved"):
    STATE.setdefault(name, set() if name == "approved" else {})


def tokens() -> dict[str, str]:
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return TOKENS["dark" if theme == "dark" else "light"]


@st.cache_resource(show_spinner="Loading the reader...")
def get_pipeline(reader: str) -> Pipeline:
    return Pipeline.from_settings(SETTINGS, reader)


def read_all(files: list[tuple[str, bytes]], reader: str) -> None:
    """Read every file not read yet with this reader; results stay in the session."""
    todo = []
    for name, data in files:
        key = f"{reader}:{hashlib.sha1(data).hexdigest()[:12]}:{name}"
        STATE.files[key] = data
        if key not in STATE.results:
            todo.append((key, name, data))
    if not todo:
        return
    try:
        pipe = get_pipeline(reader)
    except ExtractionError as exc:
        st.error(str(exc))
        return
    progress = st.progress(0.0, text="Reading...")
    for position, (key, name, data) in enumerate(todo, 1):
        progress.progress(
            (position - 1) / len(todo), text=f"Reading {name} ({position}/{len(todo)})..."
        )
        try:
            STATE.results[key] = pipe.process(Document.from_bytes(data, name))
        except (DocumentError, ExtractionError) as exc:
            st.error(f"{name}: {exc}")
    progress.empty()


def current(key: str) -> Result:
    """The result with the reviewer's corrections and approval applied."""
    result = STATE.results[key]
    receipt, review = STATE.receipts.get(key, result.receipt), STATE.reviews.get(key, result.review)
    if key in STATE.approved:  # a person signed it off; the checks stay on record
        review = review.model_copy(update={"status": "approved"})
    return Result(
        file=result.file, receipt=receipt, review=review, model=result.model,
        seconds=result.seconds, ocr_text=result.ocr_text,
    )  # fmt: skip


def status_label(key: str, result: Result) -> str:
    if key in STATE.approved:
        return "✅ Approved by reviewer"
    if result.review.status != "approved":
        return "⚠️ Needs review"
    return "✅ Corrected, checks pass" if key in STATE.receipts else "✅ Auto-approved"


def summary(keys: list[str]) -> None:
    results = [current(key) for key in keys]
    auto = sum(STATE.results[key].review.status == "approved" for key in keys)
    tiles = st.columns(4)
    tiles[0].metric("Documents", len(keys))
    tiles[1].metric("Auto-approved", auto)
    tiles[2].metric("Sent to review", len(keys) - auto)
    times = sorted(STATE.results[key].seconds for key in keys)
    tiles[3].metric("Median time per document", f"{times[len(times) // 2]:.1f} s")
    st.dataframe(
        pd.DataFrame(
            {
                "Document": [r.file for r in results],
                "Status": [status_label(k, r) for k, r in zip(keys, results, strict=True)],
                "Total": [r.receipt.total or "" for r in results],
                "Items": [len(r.receipt.items) for r in results],
                "Flagged": [", ".join(r.review.flagged) for r in results],
            }
        ),
        hide_index=True,
        width="stretch",
    )


def review_panel(key: str) -> None:
    result = current(key)
    review = STATE.reviews.get(key, result.review)
    flagged = set(review.flagged)
    version = STATE.setdefault("versions", {}).get(key, 0)  # fresh widgets after each re-check
    widget = f"{key}-{version}"
    image, panel = st.columns([2, 3], gap="large")
    with image:
        pages = Document.from_bytes(STATE.files[key], result.file).pages
        for page in pages:
            st.image(page, width="stretch")
    with panel:
        if key in STATE.approved:
            st.success("Approved by a reviewer.")
        elif review.status == "approved":
            st.success("Every check passed: ready to book.")
        else:
            st.warning(f"{len(review.failed)} check(s) failed: review the flagged fields.")
        for check in review.checks:
            icon = {"pass": "✅", "fail": "⚠️", "skip": "➖"}[check.status]
            st.markdown(f"{icon} **{check.name}**: {check.detail}")
        st.caption(f"Read by {result.model} in {result.seconds:.1f} s.")

        receipt = result.receipt
        with st.form(f"form-{widget}", border=False):
            values = {}
            columns = st.columns(4)
            for index, (field, title) in enumerate(HEADER.items()):
                values[field] = columns[index].text_input(
                    title, getattr(receipt, field) or "", key=f"{field}-{widget}"
                )
            columns = st.columns(5)
            for index, (field, title) in enumerate(AMOUNTS.items()):
                label = f"⚠️ {title}" if field in flagged else title
                values[field] = columns[index % 5].text_input(
                    label, getattr(receipt, field) or "", key=f"{field}-{widget}"
                )
            rows = pd.DataFrame(
                [
                    {
                        "check": "⚠️" if any(f.startswith(f"items[{i}]") for f in flagged) else "",
                        **{column: getattr(item, column) or "" for column in ITEM_COLUMNS},
                        "add_ons": ", ".join(sub.name for sub in item.sub_items),
                    }
                    for i, item in enumerate(receipt.items)
                ],
                columns=["check", *ITEM_COLUMNS, "add_ons"],
            )
            edited = st.data_editor(
                rows,
                num_rows="dynamic",
                hide_index=True,
                width="stretch",
                key=f"items-{widget}",
                disabled=["check", "add_ons"],
                column_config={
                    "check": st.column_config.TextColumn("", width="small"),
                    **{c: st.column_config.TextColumn(t) for c, t in ITEM_COLUMNS.items()},
                    "add_ons": st.column_config.TextColumn("Add-ons"),
                },
            )
            recheck, approve = st.columns(2)
            rechecked = recheck.form_submit_button("Re-check my corrections", width="stretch")
            approved = approve.form_submit_button("Approve", type="primary", width="stretch")
        if rechecked or approved:
            corrected = corrected_receipt(receipt, values, edited)
            STATE.receipts[key] = corrected
            STATE.reviews[key] = get_pipeline(key.split(":", 1)[0]).review(
                corrected, result.ocr_text
            )
            STATE.versions[key] = version + 1
            if approved:
                STATE.approved.add(key)
            st.rerun()


def text(value: object) -> str | None:
    """A cell or input as a stripped string; empty cells (None, NaN) become None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value).strip() or None


def corrected_receipt(original: Receipt, values: dict[str, str], items: pd.DataFrame) -> Receipt:
    lines = []
    for index, row in enumerate(items.to_dict("records")):
        name = text(row.get("name"))
        if name is None:
            continue
        subs = original.items[index].sub_items if index < len(original.items) else []
        fields = {field: text(row.get(field)) for field in ITEM_COLUMNS if field != "name"}
        lines.append(LineItem(name=name, sub_items=subs, **fields))
    fields = {field: text(value) for field, value in values.items()}
    return original.model_copy(update={**fields, "items": lines})


def benchmark_tab() -> None:
    runs = [run for run in bench.load_metrics(Path(SETTINGS.results_dir)) if run["split"] == "test"]
    if not runs:
        st.info("No benchmark results yet. Run `invoiceiq benchmark --extractor ocr-rules` first.")
        return
    st.markdown(
        f"Scored on the **CORD v2 test split**: {runs[0]['receipts']} real Indonesian receipts "
        "with labelled fields (CC BY 4.0). Field F1 follows the Donut paper's protocol, where "
        "values must match character for character; the normalized score compares amounts as "
        "numbers."
    )
    gemini = next((run for run in runs if run["extractor"] == "gemini"), None)
    if gemini:
        tiles = st.columns(4)
        tiles[0].metric("Field F1, Gemini zero-shot", f"{gemini['field_f1']:.1%}")
        tiles[1].metric("Total right", f"{gemini['total_accuracy']:.1%}")
        tiles[2].metric("Auto-approved", f"{gemini['auto_approved']:.0%}")
        if gemini.get("approved_amounts_exact") is not None:
            tiles[3].metric("Right when auto-approved", f"{gemini['approved_amounts_exact']:.1%}")
    t = tokens()
    chart = benchmark_chart(benchmark_frame(runs, bench.label), t, width=300, columns=2)
    st.altair_chart(style(chart, t), width="content")
    rates = {
        "Field F1": "field_f1", "F1 normalized": "field_f1_normalized",
        "Total right": "total_accuracy", "Every amount right": "amounts_exact",
        "Auto-approved": "auto_approved", "Right when auto-approved": "approved_amounts_exact",
        "Wrong receipts sent to review": "errors_caught",
    }  # fmt: skip
    table = pd.DataFrame(
        {
            "Extractor": [bench.label(run) for run in runs],
            **{title: [percent(run.get(key)) for run in runs] for title, key in rates.items()},
            "Median seconds": [run["median_seconds"] for run in runs],
        }
    )
    st.dataframe(table, hide_index=True, width="stretch", column_config=PERCENT_COLUMNS)
    run = st.selectbox("Per-field scores for", runs, format_func=bench.label)
    fields = pd.DataFrame(
        [
            {"Field": field, "Labels": scores["strict"]["support"],
             "F1": percent(scores["strict"]["f1"]),
             "F1 normalized": percent(scores["normalized"]["f1"])}
            for field, scores in run["fields"].items()
            if scores["strict"]["support"]
        ]
    )  # fmt: skip
    st.dataframe(fields, hide_index=True, width="stretch", column_config=PERCENT_COLUMNS)


def percent(value: float | None) -> float | None:
    return None if value is None else round(100 * value, 1)


PERCENT_COLUMNS = {
    name: st.column_config.NumberColumn(format="%.1f%%")
    for name in (
        "Field F1", "F1 normalized", "Total right", "Every amount right", "Auto-approved",
        "Right when auto-approved", "Wrong receipts sent to review", "F1",
    )
}  # fmt: skip


st.title("🧾 InvoiceIQ")
st.caption(
    "Receipts and invoices in, validated structured data out. Every result is checked against the "
    "document's own arithmetic and against local OCR, and anything that does not add up is "
    "flagged for a human."
)
with st.sidebar:
    reader_name = st.radio("Reader", list(READERS))
    reader = READERS[reader_name]
    st.caption(
        {
            "gemini": "Gemini reads the page straight into the schema. Free tier: set "
            "GEMINI_API_KEY in .env.",
            "donut": "A 2022 model trained on CORD receipts. Runs on your CPU; needs "
            '`pip install -e ".[donut]"`.',
            "ocr-rules": "The classic approach: OCR, then keyword rules. Runs on your CPU.",
        }[reader]
    )

read_tab, benchmark = st.tabs(["Read documents", "Benchmark"])
with read_tab:
    uploads = st.file_uploader(
        "Receipts or invoices: photos, scans or PDFs",
        type=[suffix.lstrip(".") for suffix in SUFFIXES],
        accept_multiple_files=True,
    )
    samples = sorted(p for p in EXAMPLES.glob("*") if p.suffix.lower() in (".jpg", ".png"))
    if samples and st.button(
        "Try the sample receipts", help="Three receipts from CORD v2 (CC BY 4.0)"
    ):
        STATE["use_samples"] = True
    files = [(upload.name, upload.getvalue()) for upload in uploads or []]
    if STATE.get("use_samples"):
        files += [(path.name, path.read_bytes()) for path in samples]
    read_all(files, reader)
    keys = [
        k
        for k in STATE.results
        if k.startswith(f"{reader}:") and k.split(":", 2)[2] in {n for n, _ in files}
    ]
    if keys:
        summary(keys)
        names = {key: current(key).file for key in keys}
        selected = st.selectbox(
            "Review", keys, format_func=lambda k: f"{names[k]}  ·  {status_label(k, current(k))}"
        )
        review_panel(selected)
        st.divider()
        results = [current(key) for key in keys]
        left, right = st.columns(2)
        left.download_button(
            "Download Excel (receipts, items, checks)", to_excel(results), "invoiceiq.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch",
        )  # fmt: skip
        right.download_button(
            "Download JSON", to_json(results), "invoiceiq.json", "application/json", width="stretch"
        )
    else:
        st.info("Upload a receipt or invoice, or try the sample receipts.")
with benchmark:
    benchmark_tab()
