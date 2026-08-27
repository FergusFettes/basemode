import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import promote_verified_models as pvm  # noqa: E402


def test_compact_scalar_arrays_collapses_string_lists() -> None:
    rendered = json.dumps({"quirks": ["no_prefill", "no_temperature"]}, indent=2)
    assert pvm._compact_scalar_arrays(rendered) == (
        '{\n  "quirks": ["no_prefill", "no_temperature"]\n}'
    )


def test_main_promotes_verified_models_not_already_known(monkeypatch, tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"models": [{"model": "openai/gpt-6-mini"}]}, indent=2) + "\n"
    )
    monkeypatch.setattr(pvm, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(
        pvm,
        "current_status",
        lambda: {
            "openai/gpt-6-mini": {"verified": True},  # already known: skip
            "deepinfra/qwen/qwen3.5-122b-a10b": {"verified": True},
            "deepinfra/qwen/qwen3.5-27b": {"verified": False},  # not verified: skip
            "groq/llama-4-scout": {"reachable": True},  # never thorough-verified
        },
    )

    assert pvm.main() == 0

    registry = json.loads(registry_path.read_text())
    models = {entry["model"]: entry for entry in registry["models"]}
    assert set(models) == {"openai/gpt-6-mini", "deepinfra/qwen/qwen3.5-122b-a10b"}
    promoted = models["deepinfra/qwen/qwen3.5-122b-a10b"]
    assert promoted["pricing_url"] == pvm._PRICING_URLS["deepinfra"]
    assert "prompt_method" not in promoted


def test_main_is_a_no_op_when_nothing_new_to_promote(monkeypatch, tmp_path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"models": []}, indent=2) + "\n")
    original = registry_path.read_text()
    monkeypatch.setattr(pvm, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(pvm, "current_status", lambda: {})

    assert pvm.main() == 0
    assert registry_path.read_text() == original
