"""Safe, no-shell execution of backend command plans."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from structbio.experiment import ExperimentPaths, update_metadata
from structbio.tools.base import CommandPlan


def execute_plan(plan: CommandPlan, paths: ExperimentPaths) -> int:
    """Run steps sequentially with argv lists and append-only logs."""

    update_metadata(paths, status="running", started_at=datetime.now(timezone.utc).isoformat())
    return_code = 0
    with paths.stdout.open("a", encoding="utf-8") as stdout, paths.stderr.open(
        "a", encoding="utf-8"
    ) as stderr:
        for step in plan.steps:
            stdout.write(f"\n[{step.name}] {step.render()}\n")
            stdout.flush()
            env = os.environ.copy()
            env.update(step.env)
            try:
                result = subprocess.run(
                    list(step.argv),
                    cwd=step.cwd,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
                return_code = result.returncode
            except OSError as exc:
                stderr.write(f"Unable to start {step.name}: {exc}\n")
                return_code = 127
            if return_code != 0:
                break
    update_metadata(
        paths,
        status="completed" if return_code == 0 else "failed",
        return_code=return_code,
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    return return_code


def plan_input_paths(plan: CommandPlan) -> list[Path]:
    values = plan.artifacts.get("absolute_input_paths", [])
    return [Path(value) for value in values]
