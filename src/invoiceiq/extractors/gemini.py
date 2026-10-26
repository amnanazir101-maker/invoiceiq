"""Gemini reads the page image straight into the Receipt schema (structured output, free tier)."""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import ValidationError

from invoiceiq.documents import Document, jpeg_bytes
from invoiceiq.extractors.base import Extraction, ExtractionError, RateLimitError
from invoiceiq.schema import Receipt

log = logging.getLogger(__name__)

# Fast, accurate on the benchmark, and with a larger free quota than the Flash models.
DEFAULT_MODEL = "gemini-3.5-flash-lite"

SYSTEM = """You read receipts, bills and invoices for a bookkeeping system and fill in the JSON \
schema you are given.

Rules:
1. Copy every value character for character as it is printed: the same digits, thousands \
separators, decimal marks, symbols (such as Rp, $, @, x, - and parentheses) and spelling. Never \
translate, reformat, correct, round or calculate a value.
2. Use null for anything that is not printed. Never work out a missing value from other values.
3. items: every purchased product or service, top to bottom, once each. A name, quantity and \
price printed on different lines belong to the same item; join a name that wraps onto the next \
line with a single space.
4. sub_items: add-ons, options, toppings, sizes or set components printed under an item (often \
indented or marked with "-", "+" or "*"). A line right under an item that costs nothing (no \
price, or a price of 0) is one of its sub_items, not a new item.
5. Subtotal, tax, service, discount, total, payment and change lines are never items, and \
neither are headings that only group items (such as "Drinks" or "Minuman").
6. quantity is the count printed for an item, with an attached "x" ("2", "1X", "1 x"). \
unit_price is the price of one unit when it is printed separately (often after "@"). price is \
the amount printed for the whole line.
7. total is the final amount due. cash, card and e_money are the amounts paid with each method; \
change is the money returned."""

PROMPT = "Extract the data from this document."

_THINKING_LEVELS = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}

# The SDK retries server errors; rate limits are handled below, per model.
_HTTP_OPTIONS = types.HttpOptions(
    timeout=180_000,  # milliseconds
    retry_options=types.HttpRetryOptions(
        attempts=2,  # one quick retry; longer waits are logged in GeminiExtractor._generate
        initial_delay=3.0,
        max_delay=30.0,
        exp_base=2.0,
        http_status_codes=[500, 502, 503, 504],
    ),
)


class GeminiExtractor:
    """Gemini with structured output: the response must parse as a `Receipt`.

    Every model has its own free quota. A per-minute limit is waited out on the same model,
    which keeps a benchmark on one model; a daily limit or an overloaded model moves the rest
    of the session to the next model in `fallback_models`.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        fallback_models: Sequence[str] = (),
        thinking_level: str = "low",
        client: genai.Client | None = None,
        api_key: str | None = None,
        max_side: int = 2048,
        rate_limit_waits: Sequence[float] = (15, 30, 60),
    ) -> None:
        self.thinking_level = thinking_level
        self.max_side = max_side
        self.rate_limit_waits = tuple(rate_limit_waits)
        try:
            self.client = client or genai.Client(
                api_key=api_key or None, http_options=_HTTP_OPTIONS
            )
        except ValueError as exc:  # raised when no API key is configured
            raise ExtractionError(
                "No Gemini API key found. Get a free key at https://aistudio.google.com/apikey "
                "and set GEMINI_API_KEY in .env (see .env.example)."
            ) from exc
        self.models = list(dict.fromkeys([model, *fallback_models]))
        self._active = 0  # index of the model new requests start with
        self._switch = threading.Lock()

    @property
    def model(self) -> str:
        return self.models[self._active]

    def extract(self, document: Document) -> Extraction:
        parts: list[Any] = [
            types.Part.from_bytes(data=jpeg_bytes(page, self.max_side), mime_type="image/jpeg")
            for page in document.pages
        ]
        started = time.perf_counter()
        response, model = self._generate([*parts, PROMPT])
        receipt = response.parsed
        if not isinstance(receipt, Receipt):
            try:
                receipt = Receipt.model_validate_json(response.text)
            except ValidationError as exc:
                raise ExtractionError(
                    f"{model} returned JSON that does not fit the schema."
                ) from exc
        usage = response.usage_metadata
        tokens = {}
        if usage is not None:
            tokens = {
                "input": usage.prompt_token_count or 0,
                "output": (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0),
            }
        return Extraction(
            receipt=receipt,
            model=model,
            seconds=round(time.perf_counter() - started, 2),
            usage=tokens,
        )

    def _generate(self, contents: list[Any]) -> tuple[Any, str]:
        index, waits = self._active, list(self.rate_limit_waits)
        while True:
            model = self.models[index]
            try:
                response = self.client.models.generate_content(
                    model=model, contents=contents, config=self._config(model)
                )
            except errors.APIError as exc:
                quota = _quota_hint(exc) if exc.code == 429 else _Quota()
                if exc.code == 429 and not quota.daily and waits:  # per-minute limit: wait it out
                    pause = min(max(quota.delay or 0, waits.pop(0)), 90)
                    log.warning("%s is rate-limited; retrying in %.0f s.", model, pause)
                    time.sleep(pause)
                    continue
                if exc.code in (429, 503) and index + 1 < len(self.models):
                    self._fall_back(
                        index,
                        "its free quota is used up" if exc.code == 429 else "it is overloaded",
                    )
                    index, waits = max(index + 1, self._active), list(self.rate_limit_waits)
                    continue
                if exc.code == 503 and waits:  # no model left to switch to: wait for this one
                    pause = waits.pop(0)
                    log.warning("%s is overloaded; retrying in %.0f s.", model, pause)
                    time.sleep(pause)
                    continue
                raise self._error(exc, quota) from exc
            self._check(response, model)
            return response, model

    def _config(self, model: str) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=Receipt,
        )
        level = _THINKING_LEVELS.get((self.thinking_level or "").lower())
        if level and model.startswith("gemini-3"):  # 2.5 models used token budgets instead
            config.thinking_config = types.ThinkingConfig(thinking_level=level)
        return config

    def _fall_back(self, index: int, reason: str) -> None:
        with self._switch:
            if self._active <= index:
                self._active = index + 1
                log.warning(
                    "%s is unavailable (%s); switching to %s.",
                    self.models[index], reason, self.models[index + 1],
                )  # fmt: skip

    def _error(self, exc: errors.APIError, quota: _Quota) -> ExtractionError:
        tried = ", ".join(self.models)
        if exc.code == 429 and quota.daily:
            size = f" ({quota.limit} requests a day)" if quota.limit else ""
            return RateLimitError(
                f"The free daily quota{size} is used up on {tried}. "
                "It resets at midnight Pacific time; every model has its own quota."
            )
        if exc.code == 429:
            return RateLimitError(
                f"Gemini's per-minute free limit was reached on {tried}. Try again in a minute."
            )
        if exc.code == 503:
            return ExtractionError(f"Gemini is overloaded ({tried}). Try again in a few minutes.")
        return ExtractionError(f"Gemini API error {exc.code}: {exc.message}")

    @staticmethod
    def _check(response: Any, model: str) -> None:
        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason:
            raise ExtractionError(f"Gemini blocked the request ({feedback.block_reason}).")
        candidate = response.candidates[0] if response.candidates else None
        reason = candidate.finish_reason if candidate else None
        if reason == types.FinishReason.MAX_TOKENS:
            raise ExtractionError(f"{model}'s answer was cut off by the output token limit.")
        if not response.text:
            raise ExtractionError(f"{model} returned an empty answer (finish reason: {reason}).")


@dataclass(frozen=True)
class _Quota:
    """What a 429 error says about the limit that was hit."""

    daily: bool = False
    delay: float | None = None  # seconds the API asks us to wait
    limit: int | None = None  # requests allowed in the quota window


def _quota_hint(exc: errors.APIError) -> _Quota:
    """Read the QuotaFailure and RetryInfo details of a 429 error."""
    details = exc.details if isinstance(exc.details, dict) else {}
    error = details.get("error", details) if isinstance(details, dict) else {}
    daily, delay, limit = False, None, None
    for item in error.get("details") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("@type", "")
        if kind.endswith("QuotaFailure"):
            for violation in item.get("violations") or []:
                if "PerDay" in str(violation.get("quotaId", "")):
                    daily = True
                    value = str(violation.get("quotaValue", ""))
                    limit = int(value) if value.isdigit() else limit
        elif kind.endswith("RetryInfo"):
            match = re.match(r"([\d.]+)s", str(item.get("retryDelay", "")))
            delay = float(match.group(1)) if match else None
    return _Quota(daily, delay, limit)
