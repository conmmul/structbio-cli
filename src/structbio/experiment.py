"""Atomic experiment creation and reproducibility records."""

from __future__ import annotations

import getpass
import json
import os
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from structbio.environment import environment_snapshot, git_commit, relevant_package_versions


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    if not cleaned:
        raise ValueError("Experiment name must contain at least one letter or number")
    return cleaned


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    config: Path
    command: Path
    metadata: Path
    environment: Path
    stdout: Path
    stderr: Path
    inputs: Path
    outputs: Path
    analysis: Path
    slurm_script: Path


class ExperimentManager:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    def candidate(self, name: str, now: datetime | None = None) -> Path:
        date = (now or datetime.now()).astimezone().strftime("%Y-%m-%d")
        prefix = f"{_safe_name(name)}_{date}"
        for index in range(1, 10_000):
            candidate = self.root / f"{prefix}_{index:03d}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Unable to allocate experiment name below {self.root}")

    def create(self, name: str) -> ExperimentPaths:
        self.root.mkdir(parents=True, exist_ok=True)
        while True:
            candidate = self.candidate(name)
            try:
                candidate.mkdir(mode=0o750)
                break
            except FileExistsError:
                continue
        paths = self.paths(candidate)
        for directory in (paths.inputs, paths.outputs, paths.analysis):
            directory.mkdir()
        paths.stdout.touch()
        paths.stderr.touch()
        return paths

    @staticmethod
    def paths(root: Path) -> ExperimentPaths:
        return ExperimentPaths(
            root=root,
            config=root / "config.yaml",
            command=root / "command.txt",
            metadata=root / "metadata.json",
            environment=root / "environment.txt",
            stdout=root / "stdout.log",
            stderr=root / "stderr.log",
            inputs=root / "inputs",
            outputs=root / "outputs",
            analysis=root / "analysis",
            slurm_script=root / "job.slurm",
        )


def write_records(
    paths: ExperimentPaths,
    *,
    config: dict[str, Any],
    command: str,
    tool_name: str,
    tool_path: Path | None,
    input_paths: list[Path],
    status: str,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    snapshot = environment_snapshot()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": paths.root.name,
        "tool": tool_name,
        "status": status,
        "created_at": now.isoformat(),
        "hostname": platform.node(),
        "user": getpass.getuser(),
        "pid": os.getpid(),
        "structbio_git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "wrapped_tool_git_commit": git_commit(tool_path) if tool_path else None,
        "python_version": snapshot["python"],
        "conda_environment": snapshot["conda_environment"],
        "cuda": snapshot["cuda"],
        "slurm_job_id": snapshot["slurm_job_id"],
        "package_versions": relevant_package_versions(),
        "input_paths": [str(path.resolve()) for path in input_paths],
        "output_paths": [str(paths.outputs.resolve())],
        "command": command,
    }
    paths.config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    paths.command.write_text(command.rstrip() + "\n", encoding="utf-8")
    paths.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    paths.environment.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return metadata


def update_metadata(paths: ExperimentPaths, **updates: Any) -> dict[str, Any]:
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    metadata.update(updates)
    paths.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
