import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import refresh_live_models as refresh  # noqa: E402

from basemode.live_models import LiveModel  # noqa: E402


def _model(model_id: str) -> LiveModel:
    return LiveModel(
        id=model_id,
        release_date="2026-08-01",
        release_date_confidence="registered",
        input_price_per_m=1.0,
        output_price_per_m=2.0,
    )


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "live_models_cache.json"
    monkeypatch.setattr(refresh, "CACHE_PATH", path)
    monkeypatch.setattr(refresh, "PROVIDER_ENDPOINTS", {"openrouter": object()})
    monkeypatch.setattr(
        refresh, "fetch_live_models", lambda provider, key: [_model("acme/one")]
    )
    monkeypatch.setattr(refresh, "dates_look_trustworthy", lambda models: True)
    return path


def _seed(path: Path, provider: str, *, age_days: float) -> None:
    stamp = (datetime.now(UTC) - timedelta(days=age_days)).isoformat()
    path.write_text(
        json.dumps(
            {
                "generated_at_utc": stamp,
                "providers": {
                    provider: {
                        "refreshed_at": stamp,
                        "reliable_dates": True,
                        "models": {"legacy/model": {"input_price_per_m": 5.0}},
                    }
                },
            }
        )
    )


def _providers(path: Path) -> dict:
    return json.loads(path.read_text())["providers"]


def test_providers_this_machine_cannot_reach_are_kept(cache: Path) -> None:
    """A refresh only reaches the providers it holds keys for.

    Replacing the file wholesale is how every Together model lost its prices:
    the scheduled job has no Together key, so it deleted what a developer's
    run had contributed.
    """
    _seed(cache, "together_ai", age_days=3)

    assert refresh.main([]) == 0

    providers = _providers(cache)
    assert set(providers) == {"openrouter", "together_ai"}
    assert (
        providers["together_ai"]["models"]["legacy/model"]["input_price_per_m"] == 5.0
    )
    assert providers["openrouter"]["models"]["acme/one"]["input_price_per_m"] == 1.0


def test_refreshed_providers_are_replaced_not_merged(cache: Path) -> None:
    """A provider that answered owns its whole entry.

    Merging model-by-model would resurrect IDs the provider has retired.
    """
    _seed(cache, "openrouter", age_days=3)

    refresh.main([])

    assert set(_providers(cache)["openrouter"]["models"]) == {"acme/one"}


def test_replace_drops_everything_this_run_could_not_reach(cache: Path) -> None:
    _seed(cache, "together_ai", age_days=3)

    refresh.main(["--replace"])

    assert set(_providers(cache)) == {"openrouter"}


def test_prune_drops_only_entries_past_the_cutoff(cache: Path) -> None:
    _seed(cache, "together_ai", age_days=200)

    refresh.main(["--prune-after-days", "90"])

    assert set(_providers(cache)) == {"openrouter"}


def test_prune_keeps_entries_inside_the_cutoff(cache: Path) -> None:
    _seed(cache, "together_ai", age_days=10)

    refresh.main(["--prune-after-days", "90"])

    assert set(_providers(cache)) == {"openrouter", "together_ai"}


def test_undated_entries_are_kept_until_pruning_is_asked_for(cache: Path) -> None:
    """Entries predate `refreshed_at`, so an absent stamp is not a verdict."""
    cache.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-01-01T00:00:00+00:00",
                "providers": {"together_ai": {"reliable_dates": True, "models": {}}},
            }
        )
    )

    refresh.main([])
    assert set(_providers(cache)) == {"openrouter", "together_ai"}

    refresh.main(["--prune-after-days", "90"])
    assert set(_providers(cache)) == {"openrouter"}


def test_refreshed_entries_carry_a_timestamp(cache: Path) -> None:
    refresh.main([])

    stamp = _providers(cache)["openrouter"]["refreshed_at"]
    assert datetime.fromisoformat(stamp).tzinfo is not None
