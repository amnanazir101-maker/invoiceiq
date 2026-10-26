"""The `invoiceiq` command: extract, benchmark, compare, serve."""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import warnings
from pathlib import Path

from rich.console import Console
from rich.table import Table

from invoiceiq.config import Settings
from invoiceiq.extractors import EXTRACTORS

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="invoiceiq", description="Receipts and invoices in, validated structured data out."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="read receipts or invoices (images or PDFs)")
    extract.add_argument("files", nargs="+", type=Path)
    extract.add_argument("--extractor", choices=EXTRACTORS)
    extract.add_argument("--model", help="model ID (Gemini by default: gemini-3.5-flash-lite)")
    extract.add_argument("--json", type=Path, help="also write the results as JSON")
    extract.add_argument(
        "--excel", type=Path, help="also write receipts, items and checks to Excel"
    )

    bench = commands.add_parser("benchmark", help="score an extractor on the CORD receipts")
    bench.add_argument("--extractor", choices=EXTRACTORS, required=True)
    bench.add_argument("--split", default="test", choices=["validation", "test"])
    bench.add_argument("--limit", type=int, help="only the first N receipts")
    bench.add_argument("--model", help="model ID (Gemini by default: gemini-3.5-flash-lite)")
    bench.add_argument(
        "--fallback", action="store_true",
        help="Gemini: switch models when one runs out of free quota (mixes models in one run)",
    )  # fmt: skip
    bench.add_argument("--run", help="results folder name (default: <extractor>-<split>)")
    bench.add_argument("--fresh", action="store_true", help="start over instead of resuming")

    compare_runs = commands.add_parser("compare", help="table of every finished benchmark run")
    compare_runs.add_argument("--markdown", action="store_true", help="print Markdown tables")
    serve = commands.add_parser("serve", help="start the REST API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # The public dataset needs no Hugging Face token; skip the hub's reminders about one.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # noisy on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    settings = Settings.from_env()

    from invoiceiq.extractors import ExtractionError

    try:
        if args.command == "extract":
            return run_extract(args, settings)
        if args.command == "benchmark":
            return run_benchmark(args, settings)
    except ExtractionError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    if args.command == "compare":
        return compare(settings, args.markdown)
    if args.command == "serve":
        import uvicorn

        from invoiceiq.api import create_app

        uvicorn.run(create_app(settings=settings), host=args.host, port=args.port)
    return 0


def run_extract(args: argparse.Namespace, settings: Settings) -> int:
    from invoiceiq.documents import Document, DocumentError
    from invoiceiq.pipeline import Pipeline

    pipeline = Pipeline.from_settings(settings, args.extractor, model=args.model)
    results = []
    for path in expand(args.files):
        try:
            document = Document.from_path(path)
        except (DocumentError, OSError) as exc:
            console.print(f"[red]{path}: {exc}[/red]")
            continue
        with console.status(f"Reading {path.name}..."):
            result = pipeline.process(document)
        results.append(result)
        show(result)
    if args.json and results:
        from invoiceiq.export import to_json

        args.json.write_text(to_json(results), encoding="utf-8")
        console.print(f"Saved {args.json}")
    if args.excel and results:
        from invoiceiq.export import to_excel

        args.excel.write_bytes(to_excel(results))
        console.print(f"Saved {args.excel}")
    return 0 if results else 1


def expand(paths: list[Path]) -> list[Path]:
    """Expand wildcards such as scans/*.pdf, which Windows shells pass through unexpanded."""
    files: list[Path] = []
    for path in paths:
        if not path.exists() and any(char in str(path) for char in "*?["):
            files += [Path(match) for match in sorted(glob.glob(str(path)))]
        else:
            files.append(path)
    return files


def show(result) -> None:
    receipt = result.receipt
    approved = result.review.status == "approved"
    badge = "[green]approved[/green]" if approved else "[yellow]needs review[/yellow]"
    console.rule(f"{result.file}  {badge}  [dim]{result.model}, {result.seconds:.1f} s[/dim]")
    items = Table(show_edge=False, header_style="bold")
    for column in ("Qty", "Item", "Unit price", "Amount"):
        items.add_column(column, justify="right" if column != "Item" else "left")
    for item in receipt.items:
        items.add_row(item.quantity or "", item.name, item.unit_price or "", item.price or "")
        for sub in item.sub_items:
            items.add_row(
                sub.quantity or "", f"  + {sub.name}", sub.unit_price or "", sub.price or ""
            )
    console.print(items)
    for name in (
        "subtotal",
        "discount",
        "service_charge",
        "tax",
        "total",
        "cash",
        "change",
        "card",
    ):
        if value := getattr(receipt, name):
            console.print(f"  {name.replace('_', ' ').capitalize():<16}{value}")
    for check in result.review.checks:
        mark = {"pass": "[green]pass[/green]", "fail": "[red]FAIL[/red]", "skip": "[dim]skip[/dim]"}
        console.print(f"  {mark[check.status]}  {check.name}: [dim]{check.detail}[/dim]")


def run_benchmark(args: argparse.Namespace, settings: Settings) -> int:
    from invoiceiq import bench
    from invoiceiq.cord import load_split
    from invoiceiq.ocr import OcrEngine
    from invoiceiq.pipeline import Pipeline

    with console.status(f"Loading the CORD {args.split} split..."):
        samples = load_split(args.split)
    samples = samples[: args.limit] if args.limit else samples
    run_name = args.run or f"{args.extractor}-{args.split}"
    out_dir = Path(settings.results_dir) / run_name
    # A disk cache, so the OCR behind the checks runs once per receipt across all runs.
    ocr = OcrEngine(cache_dir=Path(settings.data_dir) / "ocr-cache")
    fallback = args.fallback or args.extractor != "gemini"
    from invoiceiq.extractors import make_extractor

    extractor = make_extractor(
        args.extractor, settings, ocr=ocr, model=args.model, fallback=fallback
    )
    pipeline = Pipeline(extractor, ocr, settings.tolerance)

    def progress(position: int, total: int, record: dict) -> None:
        status = record.get("error") or record["status"].replace("_", " ")
        seconds = f"{record['seconds']:.1f} s" if "seconds" in record else ""
        console.print(f"  [{position}/{total}] {record['id']}  {status}  [dim]{seconds}[/dim]")

    console.print(
        f"Benchmarking [bold]{args.extractor}[/bold] on {len(samples)} receipts -> {out_dir}"
    )
    records, stopped = bench.run(pipeline, samples, out_dir, fresh=args.fresh, progress=progress)
    metrics = bench.summarize(records, samples, args.extractor, args.split)
    bench.write_metrics(out_dir, metrics)
    if stopped:
        console.print(
            f"[yellow]Stopped after {len(records)}/{len(samples)} receipts: {stopped}[/yellow]"
        )
        console.print(
            "Run the same command later to continue where it stopped. To continue today, use "
            "another model in a new run: --model <model> --run <name>."
        )
    print_metrics([metrics])
    return 0


def compare(settings: Settings, markdown: bool = False) -> int:
    from invoiceiq import bench

    runs = bench.load_metrics(Path(settings.results_dir))
    if not runs:
        console.print(
            "No finished benchmark runs yet. Try: invoiceiq benchmark --extractor ocr-rules"
        )
        return 1
    if markdown:
        print(bench.markdown(runs))
    else:
        print_metrics(runs)
    return 0


def print_metrics(runs: list[dict]) -> None:
    """Two narrow tables: how much is read correctly, and how well the checks route receipts."""
    from invoiceiq.bench import label

    runs = [metrics for metrics in runs if metrics.get("receipts")]
    if not runs:
        return
    tables = [
        ("Accuracy on CORD receipts", [
            ("Receipts", "receipts"), ("Field F1", "field_f1"),
            ("F1 normalized", "field_f1_normalized"), ("Total right", "total_accuracy"),
            ("Every amount right", "amounts_exact"), ("Median s", "median_seconds"),
        ]),
        ("Routing by the checks", [
            ("Auto-approved", "auto_approved"),
            ("Every amount right when auto-approved", "approved_amounts_exact"),
            ("Wrong receipts sent to review", "errors_caught"),
        ]),
    ]  # fmt: skip
    for title, columns in tables:
        table = Table(title=title, header_style="bold", title_justify="left")
        table.add_column("Extractor")
        for name, _ in columns:
            table.add_column(name, justify="right", max_width=16)
        for metrics in runs:
            table.add_row(label(metrics), *(_cell(key, metrics.get(key)) for _, key in columns))
        console.print(table)


def _cell(key: str, value: float | int | None) -> str:
    if value is None:
        return "-"
    if key == "receipts":
        return str(value)
    if key == "median_seconds":
        return f"{value:.1f}"
    return f"{value:.1%}"
