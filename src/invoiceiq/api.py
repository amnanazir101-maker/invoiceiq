"""REST API: POST a receipt or invoice, get validated JSON back."""

from __future__ import annotations

import threading
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from invoiceiq import __version__
from invoiceiq.config import Settings
from invoiceiq.documents import Document, DocumentError
from invoiceiq.extractors import ExtractionError, RateLimitError
from invoiceiq.pipeline import Pipeline
from invoiceiq.schema import Receipt
from invoiceiq.validation import Review

MAX_UPLOAD = 20 * 1024 * 1024  # bytes


class ExtractResponse(BaseModel):
    file: str
    status: Literal["approved", "needs_review"]
    receipt: Receipt  # values exactly as printed
    values: dict  # the same data with amounts and quantities as numbers
    review: Review
    model: str
    seconds: float


def create_app(pipeline: Pipeline | None = None, settings: Settings | None = None) -> FastAPI:
    """The app; the pipeline (and its model) is created on the first request unless given."""
    app = FastAPI(
        title="InvoiceIQ",
        version=__version__,
        description="Receipt and invoice images or PDFs in, validated structured data out.",
    )
    state: dict[str, Pipeline | None] = {"pipeline": pipeline}
    lock = threading.Lock()

    def get_pipeline() -> Pipeline:
        with lock:
            if state["pipeline"] is None:
                state["pipeline"] = Pipeline.from_settings(settings or Settings.from_env())
            return state["pipeline"]

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/extract", response_model=ExtractResponse)
    async def extract(file: UploadFile = File(...)) -> ExtractResponse:  # noqa: B008
        data = await file.read()
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"Files up to {MAX_UPLOAD // 1024 // 1024} MB are accepted.")
        try:
            document = Document.from_bytes(data, file.filename or "upload")
        except DocumentError as exc:
            raise HTTPException(415, str(exc)) from exc
        try:
            result = await run_in_threadpool(get_pipeline().process, document)
        except RateLimitError as exc:
            raise HTTPException(429, str(exc)) from exc
        except ExtractionError as exc:
            raise HTTPException(502, str(exc)) from exc
        return ExtractResponse(**result.to_dict())

    return app
