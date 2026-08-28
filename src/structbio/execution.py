"""Safe, no-shell execution of backend command plans."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

from structbio.experiment import ExperimentPaths, update_metadata
from structbio.tools.base import CommandPlan


def _pump(source: IO[str], log: IO[str], sink: IO[str] | None) -> None:
    """Copy one stream of a running tool to its log, and to the terminal."""

    for line in source:
        log.write(line)
        log.flush()
        if sink is not None:
            sink.write(line)
            sink.flush()


def execute_plan(plan: CommandPlan, paths: ExperimentPaths, *, stream: bool = True) -> int:
    """Run steps sequentially with argv lists and append-only logs.

    Output is written to the log files and, unless `stream` is false, echoed to
    the terminal as it arrives: a design job runs for hours and a researcher
    watching it should not have to tail a file to see progress.
    """

    update_metadata(paths, status="running", started_at=datetime.now(timezone.utc).isoformat())
    return_code = 0
    with paths.stdout.open("a", encoding="utf-8") as stdout, paths.stderr.open(
        "a", encoding="utf-8"
    ) as stderr:
        for step in plan.steps:
            stdout.write(f"\n[{step.name}] {step.render()}\n")
            stdout.flush()
            env = os.environ.copy()
            # Wrapped tools are Python programs; without this their output sits
            # in a block buffer until they exit and nothing appears live.
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.update(step.env)
            try:
                process = subprocess.Popen(
                    list(step.argv),
                    cwd=step.cwd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                stderr.write(f"Unable to start {step.name}: {exc}\n")
                return_code = 127
            else:
                readers = [
                    threading.Thread(
                        target=_pump,
                        args=(process.stdout, stdout, sys.stdout if stream else None),
                        daemon=True,
                    ),
                    threading.Thread(
                        target=_pump,
                        args=(process.stderr, stderr, sys.stderr if stream else None),
                        daemon=True,
                    ),
                ]
                for reader in readers:
                    reader.start()
                try:
                    return_code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    return_code = process.wait()
                    stderr.write(f"\n{step.name} was interrupted\n")
                for reader in readers:
                    reader.join()
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
