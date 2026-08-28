"""Read-only environment and installation diagnostics."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from structbio.config import ToolInstallation


def executable_path(installation: ToolInstallation) -> Path | None:
    if installation.path and installation.executable:
        candidate = installation.path / installation.executable
        if candidate.exists():
            return candidate.resolve()
    if installation.executable:
        found = shutil.which(installation.executable)
        return Path(found).resolve() if found else None
    return None


def command_output(argv: list[str], timeout: float = 3.0) -> str | None:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def git_commit(path: Path) -> str | None:
    output = command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
    return output.splitlines()[0] if output else None


def detect_gpu() -> dict[str, Any]:
    names = command_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    driver = command_output(["nvidia-smi"])
    cuda = None
    if driver and "CUDA Version:" in driver:
        cuda = driver.split("CUDA Version:", 1)[1].split()[0]
    return {
        "available": bool(names),
        "models": names.splitlines() if names else [],
        "cuda_driver": cuda,
    }


def conda_environment() -> str | None:
    return os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX")


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": platform.node(),
        "conda_environment": conda_environment(),
        "cuda": detect_gpu(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def relevant_package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    packages = ["structbio", "pydantic", "typer", "PyYAML"]
    result: dict[str, str] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def detect_unwrapped_tools() -> dict[str, str | None]:
    candidates = {
        "AlphaFold": "run_alphafold.py",
        "ColabFold": "colabfold_batch",
        "Foldseek": "foldseek",
        "MMseqs2": "mmseqs",
        "ChimeraX": "ChimeraX",
        "PyMOL": "pymol",
        "Phenix": "phenix",
        "RELION": "relion_refine",
        "EMAN2": "e2version.py",
        "DIALS": "dials.version",
        "DSSP": "mkdssp",
        "TM-align": "TMalign",
        "US-align": "USalign",
    }
    return {name: shutil.which(command) for name, command in candidates.items()}
