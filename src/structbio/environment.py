"""Read-only environment and installation diagnostics."""

from __future__ import annotations

import json
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


def command_output(
    argv: list[str], timeout: float = 3.0, *, only_on_success: bool = False
) -> str | None:
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
    if only_on_success and result.returncode:
        return None
    text = (result.stdout or result.stderr).strip()
    return text or None


def git_commit(path: Path) -> str | None:
    """Return the commit of a checkout, or None when it is not a Git checkout."""

    output = command_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], only_on_success=True
    )
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


def gpu_free_memory() -> list[tuple[int, int]]:
    """Return (index, free MiB) per GPU, most free first."""

    output = command_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        only_on_success=True,
    )
    if not output:
        return []
    entries: list[tuple[int, int]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            entries.append((int(fields[0]), int(fields[1])))
        except ValueError:
            continue
    return sorted(entries, key=lambda item: (-item[1], item[0]))


def select_idle_gpu() -> int | None:
    """Pick the GPU with the most free memory, or None without nvidia-smi."""

    entries = gpu_free_memory()
    return entries[0][0] if entries else None


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


def conda_environments() -> dict[str, Path]:
    """Map conda environment names to their prefixes, if conda is installed."""

    manager = shutil.which("conda") or shutil.which("mamba")
    if manager is None:
        return {}
    output = command_output([manager, "env", "list", "--json"], timeout=20.0, only_on_success=True)
    if not output:
        return {}
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return {}
    return {
        prefix.name: prefix
        for prefix in (Path(raw) for raw in payload.get("envs", []))
        if prefix.is_dir()
    }


def diagnose_installation(
    installation: ToolInstallation, *, tool: str, default_executable: str
) -> tuple[list[str], list[str]]:
    """Say exactly why a configured tool cannot be reached, and what to do.

    "Configured but unavailable" has several quite different causes, and a
    researcher cannot act on the difference unless it is spelled out.
    """

    problems: list[str] = []
    remedies: list[str] = []
    name = installation.executable or default_executable

    if installation.path is not None:
        path = installation.path.expanduser()
        if not path.exists():
            problems.append(f"the configured path does not exist: {path}")
            remedies.append(f"structbio install {tool} --into {path.parent}")
            remedies.append(
                "or, if it is installed elsewhere, correct the path and re-run "
                "'structbio detect'"
            )
        elif not path.is_dir():
            problems.append(f"the configured path is not a folder: {path}")
        elif not (path / name).exists():
            problems.append(f"{path} exists but does not contain {name}")
            remedies.append(
                "the checkout is incomplete or is not the right project; "
                f"see 'structbio install {tool} --dry-run'"
            )
    elif executable_path(installation) is None:
        problems.append(f"{name} is not on PATH, and no path is configured for it")
        remedies.append(f"structbio install {tool} --into ~/software")

    if installation.manager == "conda":
        if shutil.which("conda") is None and shutil.which("mamba") is None:
            problems.append("conda is not installed, so its environment cannot be used")
            remedies.append("install conda, or set manager: none if no environment is needed")
        elif installation.environment:
            environments = conda_environments()
            if environments and installation.environment not in environments:
                problems.append(
                    f"the conda environment {installation.environment!r} does not exist"
                )
                remedies.append(
                    f"create it with the steps from 'structbio install {tool} --dry-run', "
                    "or correct 'environment' in the configuration"
                )
    elif installation.manager == "pixi" and shutil.which("pixi") is None:
        problems.append("pixi is not installed")
        remedies.append("curl -fsSL https://pixi.sh/install.sh | bash")

    return problems, remedies


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
