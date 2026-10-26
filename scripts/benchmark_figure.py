"""Render docs/benchmark.png from results/*-test/metrics.json.

Needs the PNG renderer for Altair: pip install -e ".[docs]"
"""

from __future__ import annotations

from pathlib import Path

from invoiceiq import bench
from invoiceiq.charts import TOKENS, benchmark_chart, benchmark_frame, style

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    runs = [run for run in bench.load_metrics(ROOT / "results") if run["split"] == "test"]
    theme = TOKENS["light"]
    chart = benchmark_chart(benchmark_frame(runs, bench.label), theme, width=170)
    title = f"CORD v2 test split, {runs[0]['receipts']} receipts"
    chart = style(chart.properties(title=title), theme).configure_title(
        anchor="start", fontSize=14, color=theme["ink"], offset=12
    )
    out = ROOT / "docs" / "benchmark.png"
    chart.save(out, scale_factor=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
