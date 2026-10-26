"""Exports for bookkeeping: one Excel workbook (receipts, items, checks) or JSON."""

from __future__ import annotations

import io
import json

import pandas as pd

from invoiceiq.pipeline import Result
from invoiceiq.schema import RECEIPT_AMOUNTS


def tables(results: list[Result]) -> dict[str, pd.DataFrame]:
    receipts, items, checks = [], [], []
    for result in results:
        values = result.values
        receipts.append(
            {
                "file": result.file,
                "status": result.review.status,
                "merchant": values["merchant"],
                "document_number": values["document_number"],
                "date": values["date"],
                "currency": values["currency"],
                "items": len(values["items"]),
                **{name: values[name] for name in RECEIPT_AMOUNTS},
                "flagged_fields": ", ".join(result.review.flagged),
                "model": result.model,
            }
        )
        for line, item in enumerate(values["items"], 1):
            items.append(
                {
                    "file": result.file,
                    "line": line,
                    "name": item["name"],
                    "code": item["code"],
                    "quantity": item["quantity"],
                    "unit_price": item["unit_price"],
                    "discount": item["discount"],
                    "price": item["price"],
                    "add_ons": ", ".join(sub["name"] for sub in item["sub_items"]),
                }  # fmt: skip
            )
        for check in result.review.checks:
            checks.append(
                {
                    "file": result.file,
                    "check": check.name,
                    "status": check.status,
                    "detail": check.detail,
                }
            )
    return {
        "Receipts": pd.DataFrame(receipts),
        "Items": pd.DataFrame(items),
        "Checks": pd.DataFrame(checks),
    }


def to_excel(results: list[Result]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet, frame in tables(results).items():
            frame.to_excel(writer, sheet_name=sheet, index=False)
            worksheet = writer.sheets[sheet]
            worksheet.freeze_panes = "A2"
            for column in worksheet.columns:  # fit column widths to their contents
                width = max(
                    (len(str(cell.value)) for cell in column if cell.value is not None), default=8
                )
                worksheet.column_dimensions[column[0].column_letter].width = min(
                    max(width + 2, 8), 60
                )
    return buffer.getvalue()


def to_json(results: list[Result]) -> str:
    payload = [result.to_dict() for result in results]
    return json.dumps(payload[0] if len(payload) == 1 else payload, indent=2, ensure_ascii=False)
