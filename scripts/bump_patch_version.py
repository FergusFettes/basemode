#!/usr/bin/env python3
"""Bump the patch component of pyproject.toml's version. Prints the new version."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = ROOT / "pyproject.toml"

VERSION_RE = re.compile(r'^version = "(\d+)\.(\d+)\.(\d+)"$', re.MULTILINE)


def bump(text: str) -> tuple[str, str]:
    """Returns (new_text, new_version). Raises ValueError if no version line found."""
    match = VERSION_RE.search(text)
    if not match:
        raise ValueError('Could not find a `version = "X.Y.Z"` line in pyproject.toml')
    major, minor, patch = match.groups()
    new_version = f"{major}.{minor}.{int(patch) + 1}"
    new_text = VERSION_RE.sub(f'version = "{new_version}"', text, count=1)
    return new_text, new_version


def main() -> int:
    text = PYPROJECT_PATH.read_text()
    try:
        new_text, new_version = bump(text)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1

    PYPROJECT_PATH.write_text(new_text)
    subprocess.run(["uv", "lock"], cwd=ROOT, check=True)
    print(new_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
