"""The lazy `__getattr__` surface in `basemode/__init__.py` must stay in sync."""

import basemode


def test_all_matches_lazy_mapping() -> None:
    assert set(basemode.__all__) == set(basemode._LAZY)


def test_every_exported_name_is_importable() -> None:
    for name in basemode.__all__:
        assert getattr(basemode, name) is not None


def test_unknown_attribute_raises_attribute_error() -> None:
    import pytest

    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _ = basemode.nope
