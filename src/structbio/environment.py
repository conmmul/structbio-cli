"""Read-only environment and installation diagnostics."""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
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


# PyTorch publishes one wheel index per CUDA build. Taken from
# https://download.pytorch.org/whl/ on 2026-08-28; older builds are omitted
# because no current tool asks for them.
TORCH_CUDA_BUILDS: tuple[tuple[tuple[int, int], str], ...] = (
    ((11, 8), "cu118"),
    ((12, 1), "cu121"),
    ((12, 4), "cu124"),
    ((12, 6), "cu126"),
    ((12, 8), "cu128"),
    ((12, 9), "cu129"),
    ((13, 0), "cu130"),
    ((13, 2), "cu132"),
)
TORCH_INDEX = "https://download.pytorch.org/whl/"


@dataclass(frozen=True)
class TorchInstall:
    """What a PyTorch installation is, read without importing it."""

    version: str
    cuda: str | None

    @property
    def cpu_only(self) -> bool:
        return not self.cuda


def driver_cuda_version() -> tuple[int, int] | None:
    """The highest CUDA version this NVIDIA driver supports, from nvidia-smi."""

    reported = detect_gpu()["cuda_driver"]
    if not reported:
        return None
    match = re.match(r"(\d+)\.(\d+)", str(reported))
    return (int(match.group(1)), int(match.group(2))) if match else None


def select_torch_build(driver: tuple[int, int] | None) -> str:
    """Pick the newest published build the driver can run, or the CPU build.

    A CUDA build runs on any driver at least as new as it, so the right choice
    is the highest published build not above the driver's version.
    """

    if driver is None:
        return "cpu"
    usable = [name for version, name in TORCH_CUDA_BUILDS if version <= driver]
    return usable[-1] if usable else "cpu"


def torch_install_command(environment: str, build: str) -> list[str]:
    """The pip command that installs the right PyTorch into a conda environment."""

    return [
        "conda",
        "run",
        "-n",
        environment,
        "pip",
        "install",
        "torch",
        "--index-url",
        f"{TORCH_INDEX}{build}",
    ]


def find_torch(prefix: Path) -> TorchInstall | None:
    """Read a PyTorch installation's version file, without importing torch.

    Importing torch costs seconds; its `version.py` states both the release and
    the CUDA build it was compiled for, which is everything needed here.
    """

    for site in sorted(prefix.glob("lib/python3.*/site-packages/torch/version.py")):
        try:
            text = site.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        version = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)", text, re.MULTILINE)
        cuda = re.search(r"^cuda\s*(?::[^=]+)?=\s*['\"]([^'\"]+)", text, re.MULTILINE)
        if version:
            return TorchInstall(version=version.group(1), cuda=cuda.group(1) if cuda else None)
    return None


def diagnose_torch(environment: str) -> tuple[list[str], list[str]]:
    """Check the PyTorch inside a conda environment against this machine's GPU."""

    problems: list[str] = []
    remedies: list[str] = []
    prefix = conda_environments().get(environment)
    if prefix is None:
        return problems, remedies

    driver = driver_cuda_version()
    build = select_torch_build(driver)
    command = " ".join(torch_install_command(environment, build))
    torch = find_torch(prefix)

    if torch is None:
        problems.append(f"PyTorch is not installed in the conda environment {environment!r}")
        remedies.append(f"structbio fix-env, or run: {command}")
        return problems, remedies

    gpu_present = bool(detect_gpu()["available"])
    if torch.cpu_only and gpu_present:
        problems.append(
            f"PyTorch {torch.version} in {environment!r} is a CPU-only build, "
            "so this machine's GPU will not be used"
        )
        remedies.append(f"structbio fix-env --force, or run: {command}")
    elif torch.cuda and driver:
        built = tuple(int(part) for part in torch.cuda.split(".")[:2])
        if built > driver:
            problems.append(
                f"PyTorch {torch.version} in {environment!r} was built for CUDA "
                f"{torch.cuda}, but the driver supports only "
                f"{driver[0]}.{driver[1]}"
            )
            remedies.append(f"structbio fix-env --force, or run: {command}")
    return problems, remedies


def diagnose_installation(
    installation: ToolInstallation,
    *,
    tool: str,
    default_executable: str,
    needs_torch: bool = False,
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
            elif needs_torch:
                torch_problems, torch_remedies = diagnose_torch(installation.environment)
                problems.extend(torch_problems)
                remedies.extend(torch_remedies)
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
