"""Portable SLURM script generation."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

from structbio.config import ClusterProfile, ResourceConfig
from structbio.tools.base import CommandPlan


_MEMORY_RE = re.compile(r"^\d+(?:\.\d+)?[KMGTP]?(?:i?B)?$", re.IGNORECASE)
_TIME_RE = re.compile(r"^(?:\d+-)?\d{1,2}:\d{2}:\d{2}$")
_ARRAY_RE = re.compile(r"^\d+(?:-\d+(?::\d+)?)?(?:%\d+)?$")


def merged_resources(resources: ResourceConfig, profile: ClusterProfile | None) -> ResourceConfig:
    if profile is None:
        return resources
    defaults = ResourceConfig()
    merged = profile.model_dump(exclude={"name", "modules", "preamble"})
    explicit = resources.model_dump()
    for key, value in explicit.items():
        if value != getattr(defaults, key) or merged.get(key) is None:
            merged[key] = value
    return ResourceConfig.model_validate(merged)


def _validate_resource_strings(resources: ResourceConfig) -> None:
    if not _MEMORY_RE.fullmatch(resources.memory):
        raise ValueError(f"Invalid SLURM memory value: {resources.memory!r}")
    if not _TIME_RE.fullmatch(resources.time):
        raise ValueError(f"Invalid SLURM time value: {resources.time!r}")
    if resources.array and not _ARRAY_RE.fullmatch(resources.array):
        raise ValueError(f"Invalid SLURM array value: {resources.array!r}")


def generate_slurm_script(
    plan: CommandPlan,
    *,
    experiment_dir: Path,
    job_name: str,
    resources: ResourceConfig,
    profile: ClusterProfile | None = None,
) -> str:
    resources = merged_resources(resources, profile)
    _validate_resource_strings(resources)
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={resources.nodes}",
        f"#SBATCH --cpus-per-task={resources.cpus}",
        f"#SBATCH --mem={resources.memory}",
        f"#SBATCH --time={resources.time}",
        f"#SBATCH --output={experiment_dir / 'stdout.log'}",
        f"#SBATCH --error={experiment_dir / 'stderr.log'}",
    ]
    if resources.partition:
        directives.append(f"#SBATCH --partition={resources.partition}")
    if resources.account:
        directives.append(f"#SBATCH --account={resources.account}")
    if resources.qos:
        directives.append(f"#SBATCH --qos={resources.qos}")
    if resources.gpus:
        directives.append(f"#SBATCH --gpus={resources.gpus}")
    if resources.array:
        if len(plan.steps) < 2:
            raise ValueError(
                "SLURM arrays require a command plan with at least two independent steps"
            )
        directives.append(f"#SBATCH --array={resources.array}")

    body = ["", "set -euo pipefail", f"cd {shlex.quote(str(experiment_dir))}"]
    if profile:
        body.extend(f"module load {shlex.quote(module)}" for module in profile.modules)
        body.extend(profile.preamble)
    if resources.array:
        body.append('case "${SLURM_ARRAY_TASK_ID}" in')
        for index, step in enumerate(plan.steps):
            body.append(f"  {index}) {step.render()} ;;")
        body.extend(
            [
                '  *) echo "No command for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;',
                "esac",
            ]
        )
    else:
        body.extend(step.render() for step in plan.steps)
    return "\n".join(directives + body) + "\n"
