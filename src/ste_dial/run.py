"""Entry point `ste-run`: runs the agent over the grid of conditions."""

import argparse
import datetime
import io
import itertools
import json
import random
import shlex
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from ste_dial import beam, claude_cli
from ste_dial.config import Config, Subsystem, load_config, render_prompt

MAX_ATTEMPTS = 3
BACKOFF_S = (30, 120)


@dataclass(frozen=True)
class Job:
    """One cell repeat of the experiment grid."""

    subsystem: Subsystem
    level: int
    repeat: int

    @property
    def label(self) -> str:
        return f"{self.subsystem.id}/{self.level}/{self.repeat}"


class UsageLimitReached(RuntimeError):
    """The account's usage limit rejected a run; the batch cannot continue."""


def build_grid(config: Config) -> list[Job]:
    """Returns every subsystem x level x repeat job in a seeded random order.

    The shuffle interleaves the conditions in time, so a model or service
    change during the batch does not line up with one condition.
    """
    jobs = [
        Job(subsystem, level, repeat)
        for subsystem, level, repeat in itertools.product(
            config.subsystems, config.levels, range(1, config.repeats + 1)
        )
    ]
    random.Random(config.seed).shuffle(jobs)
    return jobs


def select_jobs(grid: list[Job], pilot: bool, limit: int | None) -> list[Job]:
    """Applies --pilot and --limit to the shuffled grid.

    Args:
        grid: The shuffled grid.
        pilot: Keep only the first repeat of every subsystem x level.
        limit: Keep only the first N jobs, or None for all.

    Returns:
        The jobs to run, in grid order.
    """
    jobs = [job for job in grid if job.repeat == 1] if pilot else list(grid)
    return jobs if limit is None else jobs[:limit]


def order_by_repeat(jobs: list[Job]) -> list[Job]:
    """Moves lower repeats first and keeps the shuffled order within a repeat.

    If the batch stops early, every cell then has the same number of runs,
    or differs by one.
    """
    return sorted(jobs, key=lambda job: job.repeat)


def five_hour_usage(meta: dict) -> float | None:
    """Returns the used share of the five-hour usage window, if reported."""
    windows = (meta.get("rate_limit") or {}).get("unifiedWindows") or {}
    return (windows.get("five_hour") or {}).get("utilization")


def cell_dir(results_dir: Path, job: Job) -> Path:
    """Returns the folder that holds all runs of one subsystem x level."""
    return results_dir / "raw" / job.subsystem.id / str(job.level)


def load_metas(results_dir: Path) -> list[dict]:
    """Reads the metadata of every saved run."""
    paths = sorted((results_dir / "raw").glob("*/*/*/meta.json"))
    return [json.loads(path.read_text()) for path in paths]


def is_usable(meta: dict, config: Config) -> bool:
    """Tells whether a saved run is valid and used the configured agent.

    A run made with another model or effort level is kept on disk but is not
    part of the experiment.
    """
    return bool(
        meta.get("valid")
        and meta.get("model") == config.claude.model
        and meta.get("effort") == config.claude.effort
    )


def has_valid_run(config: Config, job: Job) -> bool:
    """Tells whether a usable run of this job is already saved."""
    paths = cell_dir(config.results_dir, job).glob(f"{job.repeat}-*/meta.json")
    return any(is_usable(json.loads(path.read_text()), config) for path in paths)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _hit_usage_limit(summary: claude_cli.StreamSummary) -> bool:
    rate_limit = summary.rate_limit or {}
    return rate_limit.get("status") == "rejected"


def run_job(
    config: Config, job: Job, attempt: int, version: str, beam_head: str
) -> dict:
    """Runs the agent once and saves the transcript, answer, and metadata.

    Args:
        config: The experiment configuration.
        job: The condition to run.
        attempt: 1 for the first try, higher for retries.
        version: Output of `claude --version` for this batch.
        beam_head: SHA of the scrub commit in the Beam checkout.

    Returns:
        The metadata that was written to meta.json.

    Raises:
        UsageLimitReached: If the account's usage limit rejected the run.
    """
    run_id = f"{job.repeat}-{uuid.uuid4().hex[:8]}"
    run_dir = cell_dir(config.results_dir, job) / run_id
    run_dir.mkdir(parents=True)
    prompt = render_prompt(
        config.prompt_template, config.style_sentences, job.subsystem, job.level
    )
    command = claude_cli.build_command(config.claude, prompt)
    workdir = config.beam_workdir

    beam.reset(workdir)
    started_at = _now()
    start = time.monotonic()
    exit_code, timed_out = claude_cli.run_claude(
        command,
        workdir,
        run_dir / "transcript.jsonl",
        run_dir / "stderr.txt",
        config.claude.timeout_s,
    )
    duration_s = round(time.monotonic() - start, 1)
    finished_at = _now()
    dirty = beam.dirty_paths(workdir)
    head_after = beam.head_sha(workdir)

    lines = (run_dir / "transcript.jsonl").read_text().splitlines()
    summary = claude_cli.parse_stream(lines, str(workdir))
    result = summary.result or {}
    answer = result.get("result") or ""
    (run_dir / "answer.md").write_text(answer)

    init_problems = claude_cli.check_init(summary.init, config.claude)
    reasons = [f"init: {problem}" for problem in init_problems]
    result_ok = (
        exit_code == 0
        and result.get("subtype") == "success"
        and not result.get("is_error")
        and bool(answer.strip())
    )
    if timed_out:
        reasons.append(f"timed out after {config.claude.timeout_s} s")
    elif not result_ok:
        reasons.append(
            f"no successful result: exit code {exit_code}, "
            f"subtype {result.get('subtype')}, is_error {result.get('is_error')}"
        )
    checkout_clean = not dirty and head_after == beam_head
    if not checkout_clean:
        reasons.append(f"checkout changed: HEAD {head_after}, status {dirty}")
    if summary.touched_agent_dir:
        reasons.append("a tool call referred to .agent")
    validity = {
        "init_ok": not init_problems,
        "result_ok": result_ok,
        "checkout_clean": checkout_clean,
        "no_agent_dir_access": not summary.touched_agent_dir,
    }
    meta = {
        "run_id": run_id,
        "subsystem": asdict(job.subsystem),
        "level": job.level,
        "repeat": job.repeat,
        "attempt": attempt,
        "prompt": prompt,
        "command": command,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "duration_api_ms": result.get("duration_api_ms"),
        "turns": result.get("num_turns"),
        "cost_usd": result.get("total_cost_usd"),
        "model": config.claude.model,
        "effort": config.claude.effort,
        "model_usage": result.get("modelUsage"),
        "claude_code_version": version,
        "beam_sha": config.beam_sha,
        "beam_scrub_commit": beam_head,
        "init": claude_cli.init_fields(summary.init) if summary.init else None,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "result_subtype": result.get("subtype"),
        "permission_denials": result.get("permission_denials"),
        "tool_counts": summary.tool_counts,
        "files_read": summary.files_read,
        "paths_searched": summary.paths_searched,
        "bash_commands": summary.bash_commands,
        "rate_limit": summary.rate_limit,
        "dirty_paths": dirty,
        "validity": validity,
        "valid": all(validity.values()),
        "failure_reasons": reasons,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if not meta["valid"] and _hit_usage_limit(summary):
        raise UsageLimitReached(
            f"usage limit rejected {job.label}; rate limit info: {summary.rate_limit}"
        )
    return meta


def _print_dry_run(config: Config, jobs: list[Job]) -> None:
    for index, job in enumerate(jobs, start=1):
        prompt = render_prompt(
            config.prompt_template, config.style_sentences, job.subsystem, job.level
        )
        command = claude_cli.build_command(config.claude, prompt)
        print(f"--- [{index}/{len(jobs)}] {job.label}  (cwd: {config.beam_workdir})")
        print(prompt)
        print()
        print(shlex.join(command))
        print()


def _confirm_full_batch(config: Config, pending: int) -> bool:
    costs = [
        meta["cost_usd"]
        for meta in load_metas(config.results_dir)
        if is_usable(meta, config) and meta.get("cost_usd") is not None
    ]
    if costs:
        mean = sum(costs) / len(costs)
        print(
            f"Mean cost of the {len(costs)} valid runs so far: ${mean:.2f}. "
            f"Projected cost of the {pending} pending runs: ${mean * pending:.2f}."
        )
    else:
        print(f"No valid runs yet, so no projection for {pending} pending runs.")
    print(
        "Costs are the API-equivalent figures that Claude Code reports. On a "
        "subscription they count against the usage limits instead."
    )
    return input("Run the full batch? [y/N] ").strip().lower() == "y"


def _usage_note(meta: dict) -> str:
    windows = (meta.get("rate_limit") or {}).get("unifiedWindows") or {}
    parts = [
        f"{name} {window['utilization']:.0%}"
        for name, window in windows.items()
        if window.get("utilization") is not None
    ]
    return f" | usage: {', '.join(parts)}" if parts else ""


def main(argv: list[str] | None = None) -> int:
    """Runs the batch. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the rendered prompts and commands without running anything",
    )
    parser.add_argument(
        "--limit", type=int, help="run only the first N jobs of the shuffled grid"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="run one repeat of every subsystem x level",
    )
    parser.add_argument(
        "--yes", action="store_true", help="do not ask before a full batch"
    )
    parser.add_argument(
        "--by-repeat",
        action="store_true",
        help="finish repeat 1 of every cell before repeat 2, and so on",
    )
    parser.add_argument(
        "--stop-at-usage",
        type=float,
        help="stop when the five-hour usage window is this full (0 to 1)",
    )
    args = parser.parse_args(argv)
    # Progress lines must appear at once, also when the output is a pipe.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)

    config = load_config(args.config)
    jobs = select_jobs(build_grid(config), args.pilot, args.limit)
    if args.dry_run:
        _print_dry_run(config, jobs)
        return 0

    pending = [job for job in jobs if not has_valid_run(config, job)]
    if args.by_repeat:
        pending = order_by_repeat(pending)
    print(f"{len(jobs)} jobs selected, {len(pending)} pending.")
    if not pending:
        return 0
    is_full_batch = not args.pilot and args.limit is None
    if is_full_batch and not args.yes and not _confirm_full_batch(config, len(pending)):
        return 1

    version = claude_cli.claude_version()
    print(f"Claude Code version: {version}")
    beam_head = beam.prepare(config.beam_repo_url, config.beam_sha, config.beam_workdir)
    print(f"Beam checkout ready at scrub commit {beam_head}")

    total_cost = 0.0
    failed: list[str] = []
    for index, job in enumerate(pending, start=1):
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                meta = run_job(config, job, attempt, version, beam_head)
            except UsageLimitReached as error:
                print(f"Stopping: {error}", file=sys.stderr)
                print("Run the same command again later to resume.", file=sys.stderr)
                return 2
            total_cost += meta["cost_usd"] or 0.0
            status = "ok" if meta["valid"] else "INVALID"
            print(
                f"[{index}/{len(pending)}] {job.label} attempt {attempt}: {status}, "
                f"{meta['turns']} turns, {meta['duration_s']} s, "
                f"${meta['cost_usd'] or 0:.2f} (batch total ${total_cost:.2f})"
                f"{_usage_note(meta)}"
            )
            usage = five_hour_usage(meta)
            if (
                args.stop_at_usage is not None
                and usage is not None
                and usage >= args.stop_at_usage
            ):
                print(
                    f"Stopping: the five-hour usage window is {usage:.0%} full. "
                    f"Batch cost ${total_cost:.2f}. Run again later to resume."
                )
                return 0
            if meta["valid"]:
                break
            for reason in meta["failure_reasons"]:
                print(f"    {reason}")
            if not meta["validity"]["init_ok"]:
                print("Stopping: the isolation check failed.", file=sys.stderr)
                return 3
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S[attempt - 1])
        else:
            failed.append(job.label)
    print(f"Done. Batch cost ${total_cost:.2f}.")
    if failed:
        print(f"No valid run after {MAX_ATTEMPTS} attempts: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
