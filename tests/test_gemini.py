"""The Gemini extractor against a fake client: no network, no key."""

from types import SimpleNamespace

import pytest
from google.genai import errors, types
from PIL import Image

from invoiceiq.documents import Document
from invoiceiq.extractors.base import ExtractionError, RateLimitError
from invoiceiq.extractors.gemini import DEFAULT_MODEL, GeminiExtractor, _quota_hint
from invoiceiq.schema import Receipt

RECEIPT_JSON = (
    '{"items": [{"name": "Kopi Susu", "quantity": "2", "price": "18,000"}], "total": "18,000"}'
)


def response(text: str = RECEIPT_JSON, parsed=None):
    return SimpleNamespace(
        text=text,
        parsed=parsed,
        prompt_feedback=None,
        candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=1200, candidates_token_count=80, thoughts_token_count=40
        ),
    )


def api_error(code: int, quota: str = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"):
    """A 429 or 503 shaped like the Gemini API's, with quota details."""
    details = {
        "error": {
            "code": code,
            "message": "quota",
            "status": "RESOURCE_EXHAUSTED",
            "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
                 "violations": [{"quotaId": quota, "quotaValue": "20"}]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1s"},
            ],
        }
    }  # fmt: skip
    return errors.APIError(code, details)


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append((model, contents, config))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def extractor(outcomes, **kwargs) -> tuple[GeminiExtractor, FakeModels]:
    models = FakeModels(outcomes)
    client = SimpleNamespace(models=models)
    kwargs.setdefault("rate_limit_waits", (0, 0))
    return GeminiExtractor(client=client, **kwargs), models


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr("invoiceiq.extractors.gemini.time.sleep", lambda seconds: None)


@pytest.fixture
def document() -> Document:
    return Document.from_image(Image.new("RGB", (600, 900), "white"), "receipt.png")


def test_structured_output_becomes_a_receipt(document):
    gemini, models = extractor([response()])
    extraction = gemini.extract(document)
    assert extraction.receipt.total == "18,000"
    assert extraction.receipt.items[0].name == "Kopi Susu"
    assert extraction.model == DEFAULT_MODEL
    assert extraction.usage == {"input": 1200, "output": 120}
    _, contents, config = models.calls[0]
    assert contents[0].inline_data.mime_type == "image/jpeg"  # the page goes first
    assert config.response_schema is Receipt
    assert config.thinking_config.thinking_level == types.ThinkingLevel.LOW


def test_the_parsed_object_is_used_when_the_sdk_provides_it(document):
    parsed = Receipt(total="9.000")
    gemini, _ = extractor([response(text="{}", parsed=parsed)])
    assert gemini.extract(document).receipt is parsed


def test_a_per_minute_limit_is_waited_out_on_the_same_model(document):
    gemini, models = extractor([api_error(429), response()], fallback_models=["gemini-3.6-flash"])
    assert gemini.extract(document).model == DEFAULT_MODEL
    assert [call[0] for call in models.calls] == [DEFAULT_MODEL, DEFAULT_MODEL]


def test_a_daily_limit_moves_to_the_next_model_and_stays_there(document):
    daily = api_error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    gemini, models = extractor(
        [daily, response(), response()], fallback_models=["gemini-3.6-flash"]
    )
    assert gemini.extract(document).model == "gemini-3.6-flash"
    assert gemini.extract(document).model == "gemini-3.6-flash"  # sticky
    assert [call[0] for call in models.calls] == [
        DEFAULT_MODEL,
        "gemini-3.6-flash",
        "gemini-3.6-flash",
    ]


def test_an_overloaded_model_falls_back(document):
    gemini, _ = extractor([api_error(503), response()], fallback_models=["gemini-3.5-flash-lite"])
    assert gemini.extract(document).model == "gemini-3.5-flash-lite"


def test_an_overloaded_last_model_is_waited_for(document):
    gemini, models = extractor([api_error(503), response()])  # no fallback models
    assert gemini.extract(document).model == DEFAULT_MODEL
    assert len(models.calls) == 2


def test_running_out_of_quota_everywhere_raises_a_rate_limit_error(document):
    daily = api_error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    gemini, _ = extractor([daily])
    with pytest.raises(RateLimitError, match=r"20 requests a day"):
        gemini.extract(document)


def test_other_api_errors_are_reported(document):
    gemini, _ = extractor([errors.APIError(400, {"error": {"code": 400, "message": "bad image"}})])
    with pytest.raises(ExtractionError, match="400"):
        gemini.extract(document)


def test_invalid_json_is_an_extraction_error(document):
    gemini, _ = extractor([response(text='{"items": "not a list"}')])
    with pytest.raises(ExtractionError, match="schema"):
        gemini.extract(document)


def test_an_unreadable_item_name_does_not_lose_the_receipt(document):
    gemini, _ = extractor([response(text='{"items": [{"name": null, "price": "9.000"}]}')])
    assert gemini.extract(document).receipt.items[0].price == "9.000"


def test_thinking_level_is_only_sent_to_gemini_3_models():
    gemini, _ = extractor([], model="gemini-2.5-flash")
    assert gemini._config("gemini-2.5-flash").thinking_config is None


def test_quota_hint_reads_the_error_details():
    minute = _quota_hint(api_error(429))
    assert (minute.daily, minute.delay) == (False, 1.0)
    day = _quota_hint(api_error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert (day.daily, day.limit) == (True, 20)
