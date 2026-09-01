import json
import subprocess
import sys
from collections.abc import AsyncGenerator
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from basemode import ObservationContext, continue_text
from basemode.contributions import (
    build_bundle,
    export_bundle,
    open_contribution_pr,
    validate_bundle,
)


class _Strategy:
    name = "system"

    async def stream(self, prefix, params) -> AsyncGenerator[str, None]:
        yield " continuation"


async def test_export_is_aggregate_only_and_matches_evidence_contract(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    started = datetime.now(UTC) - timedelta(seconds=1)
    assert [
        token
        async for token in continue_text(
            "private seed",
            model="openai/example",
            observation=ObservationContext(
                source="loom",
                source_version="0.8.0",
                contribution_eligible=True,
            ),
        )
    ] == [" continuation"]
    ended = datetime.now(UTC)

    bundle = build_bundle(since=started, until=ended)
    output = export_bundle(bundle, tmp_path / f"{bundle['bundle_id']}.json")

    serialized = output.read_text()
    assert "private seed" not in serialized
    assert bundle["observations"][0]["endpoint"] == "openai/example"
    assert bundle["observations"][0]["operations"] == 1
    assert json.loads(serialized) == bundle

    evidence = Path(__file__).parents[2] / "basemode-evidence"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "basemode_evidence.cli",
            "validate",
            str(output),
            "--no-path-check",
        ],
        cwd=evidence,
        env={"PYTHONPATH": str(evidence / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


async def test_ineligible_operations_are_not_exported(monkeypatch) -> None:
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    async for _ in continue_text("private seed", model="openai/example"):
        pass

    now = datetime.now(UTC)
    try:
        build_bundle(since=now - timedelta(minutes=1), until=now)
    except ValueError as error:
        assert str(error) == "no contribution-eligible observations in window"
    else:
        raise AssertionError("ineligible operation was exported")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(recovered_operations=2),
        lambda row: row.update(successful_initial_attempts=2),
        lambda row: row.update(operations=-1),
        lambda row: row["failures"].update(not_public=1),
        lambda row: row.update(latency_ms={"count": 2, "p50": 2, "p95": 1}),
        lambda row: row.update(cost_usd=float("inf")),
    ],
)
async def test_local_validation_rejects_public_semantic_violations(
    monkeypatch, mutate
) -> None:
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    started = datetime.now(UTC) - timedelta(seconds=1)
    async for _ in continue_text(
        "private seed",
        model="openai/example",
        observation=ObservationContext(contribution_eligible=True),
    ):
        pass
    bundle = build_bundle(since=started, until=datetime.now(UTC))
    invalid = deepcopy(bundle)
    mutate(invalid["observations"][0])

    with pytest.raises(ValueError):
        validate_bundle(invalid)


async def test_pr_workflow_commits_only_the_exported_bundle(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    started = datetime.now(UTC) - timedelta(seconds=1)
    async for _ in continue_text(
        "private seed",
        model="openai/example",
        observation=ObservationContext(contribution_eligible=True),
    ):
        pass
    bundle = build_bundle(since=started, until=datetime.now(UTC))
    commands = []

    def fake_run(args, *, cwd=None, **kwargs):
        commands.append(args)
        if args[:3] == ["gh", "repo", "fork"]:
            (cwd / "basemode-evidence").mkdir()
        stdout = (
            "https://github.com/FergusFettes/basemode-evidence/pull/1\n"
            if args[:3] == ["gh", "pr", "create"]
            else ""
        )
        return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

    url = open_contribution_pr(
        bundle,
        repo="FergusFettes/basemode-evidence",
        exported_path=tmp_path / "bundle.json",
        run=fake_run,
    )

    assert url.endswith("/pull/1")
    git_add = next(command for command in commands if command[:2] == ["git", "add"])
    assert git_add[2] == "--"
    assert git_add[3].endswith(f"/{bundle['bundle_id']}.json")
