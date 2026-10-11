"""Tests for grid construction, job selection, and resuming."""

import json
from pathlib import Path

from ste_dial import run
from ste_dial.config import load_config

CONFIG = load_config(Path(__file__).resolve().parent.parent / "config.yaml")


def _save_meta(results_dir: Path, job: run.Job, run_id: str, valid: bool) -> None:
    run_dir = run.cell_dir(results_dir, job) / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"valid": valid}))


def should_build_one_job_per_subsystem_level_and_repeat():
    grid = run.build_grid(CONFIG)

    assert len(grid) == 3 * 5 * CONFIG.repeats
    assert len(set(grid)) == len(grid)


def should_shuffle_the_grid_the_same_way_for_the_same_seed():
    assert run.build_grid(CONFIG) == run.build_grid(CONFIG)


def should_not_leave_the_grid_grouped_by_subsystem():
    grid = run.build_grid(CONFIG)

    first_block = {job.subsystem.id for job in grid[: 5 * CONFIG.repeats]}

    assert len(first_block) > 1


def should_select_fifteen_first_repeats_for_the_pilot():
    jobs = run.select_jobs(run.build_grid(CONFIG), pilot=True, limit=None)

    assert len(jobs) == 15
    assert {job.repeat for job in jobs} == {1}


def should_select_the_first_jobs_of_the_shuffled_grid_with_limit():
    grid = run.build_grid(CONFIG)

    jobs = run.select_jobs(grid, pilot=False, limit=2)

    assert jobs == grid[:2]


def should_treat_job_as_pending_when_no_run_is_saved(tmp_path):
    job = run.build_grid(CONFIG)[0]

    assert not run.has_valid_run(tmp_path, job)


def should_skip_job_that_has_a_valid_run(tmp_path):
    job = run.build_grid(CONFIG)[0]
    _save_meta(tmp_path, job, f"{job.repeat}-aaaaaaaa", valid=True)

    assert run.has_valid_run(tmp_path, job)


def should_rerun_job_whose_only_saved_run_is_invalid(tmp_path):
    job = run.build_grid(CONFIG)[0]
    _save_meta(tmp_path, job, f"{job.repeat}-aaaaaaaa", valid=False)

    assert not run.has_valid_run(tmp_path, job)


def should_print_prompt_and_command_without_running_in_dry_run(capsys):
    exit_code = run.main(
        ["--config", str(CONFIG.root / "config.yaml"), "--dry-run", "--limit", "1"]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "claude --safe-mode -p" in output
    assert "You are in a checkout of Apache Beam." in output
