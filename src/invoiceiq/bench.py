"""The benchmark: run a pipeline over a CORD split and score it.

Predictions are appended to `results/<run>/predictions.jsonl` one receipt at a time, so a run
that stops (a free-tier daily limit, a closed laptop) continues where it left off. Metrics are
recomputed from that file into `metrics.json`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median

from invoiceiq.cord import Sample, to_cord
from invoiceiq.documents import Document
from invoiceiq.extractors import ExtractionError, RateLimitError
from invoiceiq.metrics import amounts_exact, score_split, total_correct
from invoiceiq.pipeline import Pipeline

# What the README and the app show, in order.
LABELS = {
    "donut": "Donut, fine-tuned on CORD",
    "ocr-rules": "OCR + rules",
}


def run(
    pipeline: Pipeline,
    samples: list[Sample],
    out_dir: Path,
    *,
    fresh: bool = False,
    progress: Callable[[int, int, dict], None] | None = None,
) -> tuple[list[dict], str | None]:
    """Process every sample not yet in the predictions file.

    Returns the records for `samples` and, if the run stopped early, why.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.jsonl"
    if fresh and path.exists():
        path.unlink()
    done = {record["id"] for record in read_records(path) if "error" not in record}  # retry errors
    todo = [sample for sample in samples if sample.id not in done]
    stopped = None
    with path.open("a", encoding="utf-8") as out:
        for position, sample in enumerate(todo, 1):
            try:
                result = pipeline.process(Document.from_image(sample.image, sample.id))
            except RateLimitError as exc:
                stopped = str(exc)
                break
            except ExtractionError as exc:  # recorded as an empty prediction: it counts as wrong
                record = {"id": sample.id, "error": str(exc), "cord": {}, "status": "needs_review"}
            else:
                record = {
                    "id": sample.id,
                    "model": result.model,
                    "seconds": result.seconds,
                    "status": result.review.status,
                    "failed_checks": [check.name for check in result.review.failed],
                    "cord": result.native if result.native is not None else to_cord(result.receipt),
                    "receipt": result.receipt.model_dump(exclude_defaults=True),
                    "usage": result.usage,
                }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            if progress:
                progress(position, len(todo), record)
    wanted = {sample.id for sample in samples}
    records = [record for record in read_records(path) if record["id"] in wanted]
    return records, stopped


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            latest[record["id"]] = record
    return list(latest.values())


def summarize(records: list[dict], samples: list[Sample], extractor: str, split: str) -> dict:
    """Metrics for every sample that has a prediction."""
    by_id = {record["id"]: record for record in records}
    rows = [(by_id[sample.id], sample.gt) for sample in samples if sample.id in by_id]
    if not rows:
        return {"extractor": extractor, "split": split, "receipts": 0}
    preds = [record.get("cord") or {} for record, _ in rows]
    gts = [gt for _, gt in rows]
    strict, normalized = score_split(preds, gts), score_split(preds, gts, normalized=True)
    pairs = list(zip(preds, gts, strict=True))
    totals = [t for t in (total_correct(p, g) for p, g in pairs) if t is not None]
    exact = [amounts_exact(p, g) for p, g in pairs]
    statuses = [record.get("status") for record, _ in rows]
    approved = [ok for ok, s in zip(exact, statuses, strict=True) if s == "approved"]
    wrong_status = [s for ok, s in zip(exact, statuses, strict=True) if not ok]
    seconds = [record["seconds"] for record, _ in rows if "seconds" in record]
    fields = sorted(
        strict.by_field, key=lambda key: -(strict.by_field[key].tp + strict.by_field[key].fn)
    )
    return {
        "extractor": extractor,
        "split": split,
        "receipts": len(rows),
        "errors": sum("error" in record for record, _ in rows),
        "models": dict(Counter(record["model"] for record, _ in rows if record.get("model"))),
        "field_f1": round(strict.overall.f1, 4),
        "precision": round(strict.overall.precision, 4),
        "recall": round(strict.overall.recall, 4),
        "field_f1_normalized": round(normalized.overall.f1, 4),
        "total_accuracy": round(mean(totals), 4) if totals else None,
        "amounts_exact": round(mean(exact), 4),
        # Routing: how many receipts the checks let through, and how often those are right.
        "auto_approved": round(len(approved) / len(rows), 4),
        "approved_amounts_exact": round(mean(approved), 4) if approved else None,
        "errors_caught": (
            round(wrong_status.count("needs_review") / len(wrong_status), 4)
            if wrong_status
            else None
        ),
        "median_seconds": round(median(seconds), 2) if seconds else None,
        "fields": {
            key: {
                "strict": strict.by_field[key].as_dict(),
                "normalized": normalized.by_field.get(key, strict.by_field[key]).as_dict(),
            }
            for key in fields
        },
    }


def write_metrics(out_dir: Path, metrics: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_metrics(results_dir: Path) -> list[dict]:
    """Every finished benchmark run, best field F1 first."""
    runs = []
    for path in sorted(Path(results_dir).glob("*/metrics.json")):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("receipts"):
            metrics["run"] = path.parent.name
            runs.append(metrics)
    return sorted(runs, key=lambda m: -m["field_f1"])


def label(metrics: dict) -> str:
    """'Gemini 3.8 Flash, zero-shot' or the label of a local extractor; other splits are named."""
    if metrics["extractor"] in LABELS:
        name = LABELS[metrics["extractor"]]
    else:
        models = metrics.get("models") or {}
        main = max(models, key=models.get) if models else metrics["extractor"]
        main = (
            main.replace("gemini-", "Gemini ")
            .replace("-flash-lite", " Flash-Lite")
            .replace("-flash", " Flash")
        )
        name = f"{main}, zero-shot"
    split = metrics.get("split", "test")
    return name if split == "test" else f"{name} ({split})"


def markdown(runs: list[dict]) -> str:
    """The README tables: accuracy, then routing by the checks."""

    def pct(value: float | None) -> str:
        return "-" if value is None else f"{value:.1%}"

    lines = [
        "| Reader | Receipts | Field F1 | F1, normalized | Total right | Every amount right "
        "| Seconds per receipt |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        seconds = "-" if run.get("median_seconds") is None else f"{run['median_seconds']:.1f}"
        lines.append(
            f"| {label(run)} | {run['receipts']} | {pct(run['field_f1'])} "
            f"| {pct(run['field_f1_normalized'])} | {pct(run['total_accuracy'])} "
            f"| {pct(run['amounts_exact'])} | {seconds} |"
        )
    lines += [
        "",
        "| Reader | Auto-approved | Every amount right when auto-approved "
        "| Wrong receipts sent to review |",
        "|---|---:|---:|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {label(run)} | {pct(run['auto_approved'])} "
            f"| {pct(run['approved_amounts_exact'])} | {pct(run['errors_caught'])} |"
        )
    return "\n".join(lines)
