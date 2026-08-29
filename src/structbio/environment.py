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


# nvidia-smi is slow to answer on a cold driver, on a busy machine, and on
# hosts with many cards. Three seconds is not enough, and a timeout here reads
# to the researcher as "you have no GPU".
NVIDIA_SMI_TIMEOUT = 20.0

# Where nvidia-smi lives when it is not on PATH, which happens in trimmed
# login environments, inside containers, and on WSL.
NVIDIA_SMI_LOCATIONS = (
    "/usr/bin/nvidia-smi",
    "/bin/nvidia-smi",
    "/usr/local/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
    "/opt/nvidia/bin/nvidia-smi",
    "/usr/lib/wsl/lib/nvidia-smi",
)

# Compute capability by GPU family, used only when the driver is too old to
# report compute_cap itself. From NVIDIA's published capabilities.
CAPABILITY_BY_NAME: tuple[tuple[str, tuple[int, int]], ...] = (
    ("h200", (9, 0)),
    ("h100", (9, 0)),
    ("gh200", (9, 0)),
    ("l40", (8, 9)),
    ("l4", (8, 9)),
    ("rtx 6000 ada", (8, 9)),
    ("rtx 5880", (8, 9)),
    ("rtx 5000 ada", (8, 9)),
    ("rtx 4500 ada", (8, 9)),
    ("rtx 4000 ada", (8, 9)),
    ("rtx 40", (8, 9)),
    ("a100", (8, 0)),
    ("a30", (8, 0)),
    ("a800", (8, 0)),
    ("a40", (8, 6)),
    ("a10", (8, 6)),
    ("a6000", (8, 6)),
    ("a5000", (8, 6)),
    ("a4500", (8, 6)),
    ("a4000", (8, 6)),
    ("a2000", (8, 6)),
    ("rtx 30", (8, 6)),
    ("v100", (7, 0)),
    ("titan v", (7, 0)),
    ("t4", (7, 5)),
    ("rtx 20", (7, 5)),
    ("quadro rtx", (7, 5)),
    ("p100", (6, 0)),
)


@dataclass(frozen=True)
class GpuReport:
    """What asking the machine about its GPUs actually produced."""

    executable: str | None
    names: tuple[str, ...] = ()
    capabilities: tuple[tuple[int, int], ...] = ()
    driver_cuda: str | None = None
    driver_version: str | None = None
    error: str | None = None
    capability_source: str = "none"

    @property
    def available(self) -> bool:
        return bool(self.names)


def nvidia_smi() -> str | None:
    """Find nvidia-smi, honouring STRUCTBIO_NVIDIA_SMI when it is somewhere odd."""

    override = os.environ.get("STRUCTBIO_NVIDIA_SMI")
    if override:
        return override if Path(override).exists() else None
    found = shutil.which("nvidia-smi")
    if found:
        return found
    for candidate in NVIDIA_SMI_LOCATIONS:
        if Path(candidate).exists():
            return candidate
    return None


def _run_nvidia_smi(executable: str, fields: str) -> tuple[str | None, str | None]:
    """Query nvidia-smi, returning its output or a description of the failure."""

    try:
        completed = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, (
            f"{executable} did not answer within {NVIDIA_SMI_TIMEOUT:.0f} seconds. "
            "The driver may be busy or wedged; try running it yourself"
        )
    except OSError as exc:
        return None, f"{executable} could not be run: {exc}"
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        return None, (
            f"{executable} exited with code {completed.returncode}"
            + (f": {detail[-1]}" if detail else "")
        )
    return completed.stdout.strip(), None


def capability_from_name(name: str) -> tuple[int, int] | None:
    """Infer a compute capability from a GPU's model name."""

    lowered = name.lower()
    for fragment, capability in CAPABILITY_BY_NAME:
        if fragment in lowered:
            return capability
    return None


def gpu_report() -> GpuReport:
    """Ask the machine about its GPUs, keeping any reason the question failed."""

    executable = nvidia_smi()
    if executable is None:
        return GpuReport(
            executable=None,
            error=(
                "nvidia-smi was not found on PATH or in the usual locations. "
                "Set STRUCTBIO_NVIDIA_SMI to its full path if it is installed "
                "somewhere else"
            ),
        )

    output, error = _run_nvidia_smi(executable, "name,driver_version")
    if output is None:
        return GpuReport(executable=executable, error=error)
    names: list[str] = []
    driver_version: str | None = None
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if fields and fields[0]:
            names.append(fields[0])
            if len(fields) > 1 and not driver_version:
                driver_version = fields[1]
    if not names:
        return GpuReport(
            executable=executable,
            driver_version=driver_version,
            error=f"{executable} ran but listed no GPUs",
        )

    # compute_cap needs a reasonably recent driver; fall back to the model name
    # rather than treating an old driver as an absent card.
    capabilities: list[tuple[int, int]] = []
    source = "none"
    reported, _ = _run_nvidia_smi(executable, "compute_cap")
    if reported:
        for line in reported.splitlines():
            match = re.match(r"\s*(\d+)\.(\d+)", line)
            if match:
                capabilities.append((int(match.group(1)), int(match.group(2))))
        if capabilities:
            source = "nvidia-smi"
    if not capabilities:
        for name in names:
            inferred = capability_from_name(name)
            if inferred:
                capabilities.append(inferred)
        if capabilities:
            source = "model name"

    driver_cuda = None
    full = command_output([executable], timeout=NVIDIA_SMI_TIMEOUT)
    if full and "CUDA Version:" in full:
        driver_cuda = full.split("CUDA Version:", 1)[1].split()[0]

    return GpuReport(
        executable=executable,
        names=tuple(names),
        capabilities=tuple(capabilities),
        driver_cuda=driver_cuda,
        driver_version=driver_version,
        capability_source=source,
    )


def detect_gpu() -> dict[str, Any]:
    report = gpu_report()
    return {
        "available": report.available,
        "models": list(report.names),
        "cuda_driver": report.driver_cuda,
        "error": report.error,
    }


def gpu_free_memory() -> list[tuple[int, int]]:
    """Return (index, free MiB) per GPU, most free first."""

    executable = nvidia_smi()
    if executable is None:
        return []
    output = command_output(
        [executable, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        timeout=NVIDIA_SMI_TIMEOUT,
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
class PinnedEnvironment:
    """An environment whose versions an upstream file fixes.

    Upgrading PyTorch inside one of these breaks the tool: the rest of the
    environment was solved against the pinned version. The remedy has to repair
    the environment as the project defines it, never install a newer wheel.
    """

    file: str
    pins: str
    repair: str
    note: str
    cuda: str = ""

    def remedies(self, environment: str) -> list[str]:
        return [
            f"{self.file} pins {self.pins}, so do not install a newer PyTorch here: "
            f"{self.note}",
            f"repair it with: {self.repair.format(environment=environment)}",
            f"or rebuild it: conda env remove -n {environment}, then re-create it from "
            f"{self.file}",
        ]


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


def _site_packages(prefix: Path) -> list[Path]:
    """Where this environment's own interpreter imports from, most likely first.

    Globbing lib/python3.* and taking the first match sorts python3.10 ahead of
    python3.9, so an environment left with a stale directory answers about a
    PyTorch it would never import. Ask the interpreter instead, and only fall
    back to guessing when it cannot be run.
    """

    for relative in ("bin/python", "bin/python3", "python.exe"):
        interpreter = prefix / relative
        if not interpreter.exists():
            continue
        output = command_output(
            [
                str(interpreter),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            timeout=60.0,
            only_on_success=True,
        )
        if output:
            path = Path(output.splitlines()[-1].strip())
            if path.is_dir():
                return [path]
    return [
        candidate.parent
        for candidate in sorted(prefix.glob("lib/python3.*/site-packages/torch"))
    ]


def find_torch(prefix: Path) -> TorchInstall | None:
    """Read a PyTorch installation's version file, without importing torch.

    Importing torch costs seconds; its `version.py` states both the release and
    the CUDA build it was compiled for, which is everything needed here.
    """

    for packages in _site_packages(prefix):
        site = packages / "torch" / "version.py"
        if not site.is_file():
            continue
        try:
            text = site.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue  # unreadable; try the next location rather than guessing
        version = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)", text, re.MULTILINE)
        cuda = re.search(r"^cuda\s*(?::[^=]+)?=\s*['\"]([^'\"]+)", text, re.MULTILINE)
        if version:
            return TorchInstall(version=version.group(1), cuda=cuda.group(1) if cuda else None)
    return None


# The oldest CUDA toolkit that emits code for each GPU architecture. A PyTorch
# built against an older toolkit has no kernels for the card and fails with
# "no kernel image is available for execution on the device", which no amount
# of reinstalling at the same pinned version can fix.
MINIMUM_CUDA_FOR_CAPABILITY: tuple[tuple[tuple[int, int], tuple[int, int], str], ...] = (
    ((12, 0), (12, 8), "Blackwell, such as the RTX 50 series"),
    ((10, 0), (12, 8), "Blackwell datacenter"),
    ((9, 0), (11, 8), "Hopper"),
    ((8, 9), (11, 8), "Ada Lovelace, such as the RTX 40 series and L40"),
    ((8, 6), (11, 1), "Ampere"),
    ((8, 0), (11, 0), "Ampere datacenter"),
)


def gpu_capabilities() -> list[tuple[int, int]]:
    """Each GPU's compute capability, from the driver or from its model name."""

    return list(gpu_report().capabilities)


def required_cuda(capability: tuple[int, int]) -> tuple[tuple[int, int], str] | None:
    """The oldest CUDA that supports a compute capability, and its architecture."""

    for supported, cuda, architecture in MINIMUM_CUDA_FOR_CAPABILITY:
        if capability >= supported:
            return cuda, architecture
    return None


def unsupported_by(cuda: str | tuple[int, int]) -> tuple[tuple[int, int], str] | None:
    """Return the GPU requirement a given CUDA version cannot meet, if any.

    Checks the newest card in the machine: if a CUDA build cannot address it,
    that card sits idle or the run fails outright.
    """

    built = (
        tuple(int(part) for part in cuda.split(".")[:2]) if isinstance(cuda, str) else cuda
    )
    capabilities = gpu_capabilities()
    if not capabilities:
        return None
    requirement = required_cuda(max(capabilities))
    if requirement is None:
        return None
    needed, architecture = requirement
    return (needed, architecture) if built < needed else None


def diagnose_torch(
    environment: str,
    *,
    pinned: PinnedEnvironment | None = None,
    tool_name: str = "rfdiffusion",
) -> tuple[list[str], list[str], list[str]]:
    """Check a conda environment's PyTorch. Returns problems, warnings, remedies.

    A missing or undersupported build stops a run; a CPU-only build only makes
    it slow, so that is a warning and the researcher decides.
    """

    problems: list[str] = []
    warnings: list[str] = []
    remedies: list[str] = []
    prefix = conda_environments().get(environment)
    if prefix is None:
        return problems, warnings, remedies

    driver = driver_cuda_version()
    build = select_torch_build(driver)
    upgrade = " ".join(torch_install_command(environment, build))
    torch = find_torch(prefix)

    if torch is None:
        problems.append(f"PyTorch is not installed in the conda environment {environment!r}")
        remedies.extend(
            pinned.remedies(environment) if pinned else [f"structbio fix-env, or run: {upgrade}"]
        )
        return problems, warnings, remedies

    gpu_present = bool(detect_gpu()["available"])
    if pinned and pinned.cuda and gpu_present:
        blocked = unsupported_by(pinned.cuda)
        if blocked is not None:
            needed, architecture = blocked
            # A warning, not a verdict. This reads a file rather than running
            # anything, and an environment that works is not structbio's to
            # declare broken; 'structbio env verify' settles it by computing on
            # the card.
            warnings.append(
                f"this machine's GPU is {architecture}, which needs CUDA "
                f"{needed[0]}.{needed[1]} or newer, while {pinned.file} pins CUDA "
                f"{pinned.cuda}. If the GPU is unused or a run fails with "
                "'no kernel image is available', that is why"
            )
            remedies.append(
                f"check it for certain with: structbio env verify {tool_name}"
            )
            return problems, warnings, remedies

    if torch.cpu_only and gpu_present:
        warnings.append(
            f"PyTorch {torch.version} in {environment!r} is a CPU-only build, so this "
            "machine's GPU will not be used and runs will be very slow"
        )
        remedies.extend(
            pinned.remedies(environment)
            if pinned
            else [f"structbio fix-env --force, or run: {upgrade}"]
        )
    elif torch.cuda and driver:
        blocked = unsupported_by(torch.cuda)
        if blocked is not None:
            needed, architecture = blocked
            problems.append(
                f"PyTorch {torch.version} in {environment!r} was built for CUDA "
                f"{torch.cuda}, which has no kernels for this machine's GPU "
                f"({architecture}, needing CUDA {needed[0]}.{needed[1]} or newer)"
            )
            remedies.append(f"structbio fix-env --force, or run: {upgrade}")
            return problems, warnings, remedies
        built = tuple(int(part) for part in torch.cuda.split(".")[:2])
        if built > driver:
            problems.append(
                f"PyTorch {torch.version} in {environment!r} was built for CUDA "
                f"{torch.cuda}, which this driver ({driver[0]}.{driver[1]}) cannot run"
            )
            remedies.extend(
                pinned.remedies(environment)
                if pinned
                else [f"structbio fix-env --force, or run: {upgrade}"]
            )
    return problems, warnings, remedies


def diagnose_installation(
    installation: ToolInstallation,
    *,
    tool: str,
    default_executable: str,
    needs_torch: bool = False,
    pinned: PinnedEnvironment | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Say why a configured tool cannot be reached, and what to do about it.

    Returns problems that stop a run, warnings that only degrade it, and the
    remedies for both. "Configured but unavailable" has several quite different
    causes, and a researcher cannot act on the difference unless it is spelled
    out.
    """

    problems: list[str] = []
    warnings: list[str] = []
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
                found, degraded, fixes = diagnose_torch(
                    installation.environment, pinned=pinned, tool_name=tool
                )
                problems.extend(found)
                warnings.extend(degraded)
                remedies.extend(fixes)
    elif installation.manager == "pixi" and shutil.which("pixi") is None:
        problems.append("pixi is not installed")
        remedies.append("curl -fsSL https://pixi.sh/install.sh | bash")

    return problems, warnings, remedies


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
