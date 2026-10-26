"""The public-demo helpers: replaying saved sample results and the daily request cap."""

from datetime import date
from pathlib import Path

from invoiceiq.demo import DailyQuota, load_samples, replay

ROOT = Path(__file__).resolve().parent.parent


def test_the_daily_quota_stops_at_its_limit_and_resets_the_next_day():
    today = [date(2026, 10, 26)]
    quota = DailyQuota(2, today=lambda: today[0])
    assert [quota.take(), quota.take(), quota.take()] == [True, True, False]
    today[0] = date(2026, 10, 27)
    assert quota.take() and quota.used == 1


def test_a_zero_limit_means_no_limit():
    quota = DailyQuota(0)
    assert all(quota.take() for _ in range(100))


def test_saved_samples_replay_with_the_checks_run_again():
    samples = load_samples(ROOT / "examples" / "samples.json")
    assert set(samples) == {
        "receipt-add-ons.jpg", "receipt-faded.jpg", "receipt-two-line-items.jpg",
    }  # fmt: skip
    clean = replay("receipt-add-ons.jpg", samples["receipt-add-ons.jpg"])
    assert clean.review.status == "approved" and clean.receipt.total == "70,000"
    hidden = replay("receipt-two-line-items.jpg", samples["receipt-two-line-items.jpg"])
    assert hidden.review.flagged == ["items[1].price"]  # inferred, not printed: sent to review


def test_missing_samples_are_an_empty_dict(tmp_path):
    assert load_samples(tmp_path / "nope.json") == {}
