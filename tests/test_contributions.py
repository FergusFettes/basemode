import json
import subprocess
import sys
from collections.abc import AsyncGenerator
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from basemode import ObservationContext, continue_text, observations
from basemode.cli import app as cli_app
from basemode.contributions import (
    build_bundle,
    build_contribution,
    export_bundle,
    open_contribution_pr,
    release_bundle,
    validate_bundle,
)


class _Strategy:
    name = "system"

    async def stream(self, prefix, params) -> AsyncGenerator[str, None]:
        yield " continuation"


async def test_export_is_aggregate_only(monkeypatch, tmp_path: Path) -> None:
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


async def test_export_matches_sibling_evidence_contract(
    monkeypatch, tmp_path: Path
) -> None:
    evidence = Path(__file__).parents[2] / "basemode-evidence"
    if not evidence.is_dir():
        pytest.skip("requires a sibling basemode-evidence checkout")

    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    started = datetime.now(UTC) - timedelta(seconds=1)
    async for _ in continue_text(
        "private seed",
        model="openai/example",
        observation=ObservationContext(),
    ):
        pass
    bundle = build_bundle(since=started, until=datetime.now(UTC))
    output = export_bundle(bundle, tmp_path / f"{bundle['bundle_id']}.json")

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


async def test_existing_operations_are_exportable_without_prior_opt_in(
    monkeypatch,
) -> None:
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    async for _ in continue_text("private seed", model="openai/example"):
        pass

    now = datetime.now(UTC)
    bundle = build_bundle(since=now - timedelta(minutes=1), until=now)

    assert bundle["observations"][0]["operations"] == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(recovered_operations=2),
        lambda row: row.update(successful_initial_attempts=2),
        lambda row: row.update(operations=-1),
        lambda row: row["failures"].update(not_public=1),
        lambda row: row.update(latency_ms={"count": 2, "p50": 2, "p95": 1}),
        lambda row: row.update(latency_ms={"count": 9, "p50": 1, "p95": 2}),
        lambda row: row.update(ttft_ms={"count": 9, "p50": 1, "p95": 2}),
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
        observation=ObservationContext(),
    ):
        pass
    bundle = build_bundle(since=started, until=datetime.now(UTC))
    invalid = deepcopy(bundle)
    mutate(invalid["observations"][0])

    with pytest.raises(ValueError):
        validate_bundle(invalid)


async def test_ttft_is_bounded_by_attempts_not_operations(monkeypatch) -> None:
    """A logical operation can carry more than one token-producing request.

    A resumed verification probe re-runs a configuration that already
    succeeded, so its operation ends up with two successful attempts and two
    TTFT samples. Bounding TTFT by successful operations rejected that as
    invalid — real observations that were never wrong.
    """
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    started = datetime.now(UTC) - timedelta(seconds=1)
    async for _ in continue_text("private seed", model="openai/example"):
        pass

    bundle = build_bundle(since=started, until=datetime.now(UTC))
    row = bundle["observations"][0]
    row["attempts"] += 1
    row["ttft_ms"] = {"count": row["successful_operations"] + 1, "p50": 10, "p95": 20}

    validate_bundle(bundle)


async def test_pr_workflow_commits_only_the_exported_bundle(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    started = datetime.now(UTC) - timedelta(seconds=1)
    async for _ in continue_text(
        "private seed",
        model="openai/example",
        observation=ObservationContext(),
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


async def test_contribution_window_accepts_the_documented_z_suffix(monkeypatch) -> None:
    """Every timestamp the docs and the ledger show ends in `Z`.

    Typer's own datetime option takes a fixed list of formats that excludes
    it, so the documented example was rejected before it reached the window.
    """
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    async for _ in continue_text("private seed", model="openai/example"):
        pass
    started = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = CliRunner().invoke(cli_app, ["contribute", "preview", "--since", started])

    assert result.exit_code == 0, result.output
    assert '"schema_version": 1' in result.output


def test_contribution_window_rejects_unparseable_timestamps() -> None:
    result = CliRunner().invoke(
        cli_app, ["contribute", "preview", "--since", "not-a-date"]
    )

    assert result.exit_code == 2
    assert "ISO-8601" in result.output


async def _record_one_operation(monkeypatch) -> None:
    monkeypatch.setattr("basemode.continue_.detect_strategy", lambda *args: _Strategy())
    async for _ in continue_text("private seed", model="openai/example"):
        pass


async def test_exported_operations_are_not_counted_twice(
    monkeypatch, tmp_path: Path
) -> None:
    """Windows overlap freely, so submission is tracked per operation."""
    await _record_one_operation(monkeypatch)
    started = datetime.now(UTC) - timedelta(days=1)

    first = build_contribution(since=started, until=datetime.now(UTC))
    assert first.bundle["observations"][0]["operations"] == 1
    export_bundle(first, tmp_path / "first.json")

    # The same window, and a wider one, now have nothing left to contribute.
    for since in (started, started - timedelta(days=7)):
        with pytest.raises(ValueError, match="no unsubmitted observations"):
            build_contribution(since=since, until=datetime.now(UTC))


async def test_a_new_operation_after_an_export_is_still_contributable(
    monkeypatch, tmp_path: Path
) -> None:
    await _record_one_operation(monkeypatch)
    started = datetime.now(UTC) - timedelta(days=1)
    export_bundle(
        build_contribution(since=started, until=datetime.now(UTC)),
        tmp_path / "first.json",
    )

    await _record_one_operation(monkeypatch)
    second = build_contribution(since=started, until=datetime.now(UTC))

    assert second.bundle["observations"][0]["operations"] == 1
    assert second.operation_ids != ()


async def test_releasing_an_unsubmitted_export_frees_only_its_operations(
    monkeypatch, tmp_path: Path
) -> None:
    await _record_one_operation(monkeypatch)
    started = datetime.now(UTC) - timedelta(days=1)
    first = build_contribution(since=started, until=datetime.now(UTC))
    export_bundle(first, tmp_path / "first.json")
    await _record_one_operation(monkeypatch)
    second = build_contribution(since=started, until=datetime.now(UTC))
    export_bundle(second, tmp_path / "second.json")

    released = release_bundle(first.bundle["bundle_id"])

    assert released == len(first.operation_ids) == 1
    again = build_contribution(since=started, until=datetime.now(UTC))
    assert again.operation_ids == first.operation_ids


async def test_releasing_a_submitted_bundle_is_refused(
    monkeypatch, tmp_path: Path
) -> None:
    await _record_one_operation(monkeypatch)
    contribution = build_contribution(
        since=datetime.now(UTC) - timedelta(days=1), until=datetime.now(UTC)
    )
    export_bundle(contribution, tmp_path / "bundle.json")
    with observations._db() as conn:
        conn.execute(
            "UPDATE contribution_batches SET status='submitted' WHERE bundle_id=?",
            (contribution.bundle["bundle_id"],),
        )

    with pytest.raises(ValueError, match="submitted upstream"):
        release_bundle(contribution.bundle["bundle_id"])


def test_releasing_an_unknown_bundle_is_refused() -> None:
    with pytest.raises(ValueError, match="no local record"):
        release_bundle("00000000-0000-0000-0000-000000000000")
