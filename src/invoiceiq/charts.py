"""Altair charts for the app and the README figure (kept here so they can be tested and rendered).

Colors come from a validated reference palette. The benchmark is an emphasis chart, not a
categorical one: the accent hue marks the extractor the page is about (Gemini) and gray marks
the baselines, so no legend is needed and each bar is labelled directly.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# Marks have at least 3.5:1 contrast and text at least 5:1 against both surfaces.
TOKENS = {
    "light": {"accent": "#2a78d6", "context": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "muted": "#6f6d66", "ink": "#0b0b0b", "surface": "#fcfcfb"},
    "dark": {"accent": "#3987e5", "context": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "muted": "#a3a19a", "ink": "#f0efec", "surface": "#1a1a19"},
}  # fmt: skip

METRICS = {
    "field_f1": "Field F1 (strict)",
    "field_f1_normalized": "Field F1 (normalized)",
    "total_accuracy": "Total right",
    "amounts_exact": "Every amount right",
}


def style(chart: alt.TopLevelMixin, t: dict[str, str]) -> alt.TopLevelMixin:
    """Recessive hairline axes and grid; text in ink tokens, never in series colors."""
    return (
        chart.configure_axis(
            gridColor=t["grid"], gridWidth=1, domainColor=t["axis"], tickColor=t["axis"],
            labelColor=t["muted"], titleColor=t["muted"], labelFontSize=12, titleFontSize=12,
        )
        .configure_view(strokeWidth=0)
        .configure_header(labelColor=t["ink"], labelFontSize=13, labelFontWeight="bold",
                          labelAnchor="start", title=None)
        .configure(background=t["surface"])
    )  # fmt: skip


def benchmark_frame(runs: list[dict], label) -> pd.DataFrame:
    rows = []
    for run in runs:
        for key, title in METRICS.items():
            if run.get(key) is not None:
                rows.append(
                    {"extractor": label(run), "metric": title, "score": run[key],
                     "highlight": run["extractor"] == "gemini", "receipts": run["receipts"]}
                )  # fmt: skip
    return pd.DataFrame(rows)


def benchmark_chart(
    frame: pd.DataFrame, t: dict[str, str], width: int = 200, columns: int = 4
) -> alt.FacetChart:
    """One small bar chart per metric (`columns` per row), extractors on the y axis, best first."""
    order = (
        frame[frame["metric"] == METRICS["field_f1"]]
        .sort_values("score", ascending=False)["extractor"]
        .tolist()
    )
    base = alt.Chart(frame).encode(
        y=alt.Y("extractor:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("score:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", tickCount=3, grid=True)),
        tooltip=[alt.Tooltip("extractor:N", title="Extractor"),
                 alt.Tooltip("metric:N", title="Metric"),
                 alt.Tooltip("score:Q", title="Score", format=".1%"),
                 alt.Tooltip("receipts:Q", title="Receipts")],
    )  # fmt: skip
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.condition(alt.datum.highlight, alt.value(t["accent"]), alt.value(t["context"]))
    )
    labels = base.mark_text(align="left", dx=6, fontSize=12, color=t["ink"]).encode(
        text=alt.Text("score:Q", format=".1%")
    )
    return (
        alt.layer(bars, labels)
        .properties(width=width, height=alt.Step(30))
        .facet(
            facet=alt.Facet("metric:N", sort=list(METRICS.values()), title=None),
            columns=columns,
            spacing={"column": 48, "row": 24},
        )
        .resolve_scale(y="shared")
    )
