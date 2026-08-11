import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import bump_patch_version as bpv  # noqa: E402


def test_bump_increments_patch_only() -> None:
    text = '[project]\nname = "basemode"\nversion = "0.1.5"\ndescription = "x"\n'
    new_text, new_version = bpv.bump(text)
    assert new_version == "0.1.6"
    assert 'version = "0.1.6"' in new_text
    assert 'version = "0.1.5"' not in new_text


def test_bump_only_touches_the_first_version_line() -> None:
    text = 'version = "1.2.3"\ndependencies = ["foo>=2.0.0"]\n'
    new_text, new_version = bpv.bump(text)
    assert new_version == "1.2.4"
    assert 'dependencies = ["foo>=2.0.0"]' in new_text


def test_bump_raises_when_no_version_line() -> None:
    with pytest.raises(ValueError):
        bpv.bump('[project]\nname = "basemode"\n')
