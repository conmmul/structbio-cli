"""Build and check the environments the wrapped tools need.

Version matching is the part of this software that wastes researchers' days, so
structbio does it from what the machine reports rather than leaving it to each
person. Two rules keep that honest:

- a plan is only offered when the versions involved can actually work together,
  and where they cannot the reason is stated instead;
- every environment is proved by running code in it, not by assuming the
  install succeeded.

Facts below were read from upstream on 2026-08-28: RFdiffusion's env/SE3nv.yml
and setup.py, its bundled SE3Transformer requirements, DGL's published wheel
indexes, and PyTorch's wheel index list.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from structbio import environment
from structbio.config import ToolInstallation


# Pairings where PyTorch and DGL both publish a wheel for the same CUDA and the
# same PyTorch release. Verified on 2026-08-28 against
# download.pytorch.org/whl/ and data.dgl.ai/wheels/torch-2.3/. DGL builds cu124
# too, but not against PyTorch 2.3, and PyTorch 2.3.1 has no cu124 wheel, so
# that combination does not exist and is deliberately not offered.
TORCH_RELEASE = "2.3.*"
# The Python versions both PyTorch 2.3 and DGL publish CUDA wheels for, from
# their indexes on 2026-08-28. An environment outside this range finds nothing.
SUPPORTED_PYTHON_TAGS = ("cp38", "cp39", "cp310", "cp311", "cp312")
DGL_TORCH_INDEX = "2.3"
SUPPORTED_PAIRINGS: tuple[tuple[tuple[int, int], str], ...] = (
    ((11, 8), "cu118"),
    ((12, 1), "cu121"),
)
DGL_INDEX = "https://data.dgl.ai/wheels/torch-{torch}/{build}/repo.html"

# The bundled SE3Transformer's own requirements.txt.
SE3_REQUIREMENTS = ("e3nn==0.3.3", "wandb", "pynvml", "decorator")
RFDIFFUSION_EXTRAS = ("hydra-core", "pyrsistent", "icecream", "opt_einsum")


@dataclass(frozen=True)
class Step:
    description: str
    argv: tuple[str, ...]

    def render(self) -> str:
        return " ".join(self.argv)


@dataclass(frozen=True)
class EnvironmentPlan:
    """How to build one tool's environment on this particular machine."""

    tool: str
    environment: str
    steps: tuple[Step, ...] = ()
    upstream_verified: bool = True
    notes: tuple[str, ...] = ()
    blocked: str | None = None
    alternatives: tuple[str, ...] = ()

    @property
    def possible(self) -> bool:
        return self.blocked is None and bool(self.steps)


@dataclass
class ProbeResult:
    """What running code inside an environment actually reported."""

    ok: bool
    values: dict[str, object] = field(default_factory=dict)
    error: str | None = None

    def failures(self) -> list[str]:
        """The checks that failed, in the order a researcher should read them."""

        if self.error:
            return [self.error]
        problems: list[str] = []
        values = self.values
        if "torch" not in values:
            problems.append(
                "PyTorch could not be imported: " + str(values.get("torch_error", "unknown"))
            )
            return problems
        if not values.get("cuda_available"):
            problems.append(
                f"PyTorch {values['torch']} is installed but reports no usable GPU "
                f"(built for CUDA {values.get('torch_cuda') or 'none'})"
            )
        elif not values.get("gpu_allocation"):
            problems.append(
                "PyTorch sees the GPU but cannot compute on it: "
                + str(values.get("gpu_error", "unknown"))
                + ". This is the signature of a build with no kernels for this card"
            )
        for module in ("dgl", "se3_transformer", "hydra", "numpy"):
            if module in values and values[module] is False:
                problems.append(
                    f"{module} could not be imported: "
                    + str(values.get(f"{module}_error", "unknown"))
                )
        return problems

    def summary(self) -> str:
        values = self.values
        if "torch" not in values:
            return "PyTorch is not usable in this environment"
        device = values.get("device") or "no GPU"
        return (
            f"PyTorch {values['torch']} (CUDA {values.get('torch_cuda') or 'none'}), "
            f"device: {device}"
        )


PROBE = '''
import json, sys
result = {"python": sys.version.split()[0]}
try:
    import torch
    result["torch"] = torch.__version__
    result["torch_cuda"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    if result["cuda_available"]:
        result["device"] = torch.cuda.get_device_name(0)
        try:
            # Allocating and computing is what catches a build with no kernels
            # for this card; is_available() alone returns True regardless.
            (torch.zeros(64, device="cuda") + 1).sum().item()
            result["gpu_allocation"] = True
        except Exception as exc:
            result["gpu_allocation"] = False
            result["gpu_error"] = str(exc)[:400]
except Exception as exc:
    result["torch_error"] = str(exc)[:400]
for module in __STRUCTBIO_MODULES__:
    try:
        __import__(module)
        result[module] = True
    except Exception as exc:
        result[module] = False
        result[module + "_error"] = str(exc)[:200]
print("STRUCTBIO_PROBE " + json.dumps(result))
'''

PROBE_MODULES = {
    "rfdiffusion": ("dgl", "se3_transformer", "hydra"),
    "proteinmpnn": ("numpy",),
}


def probe_source(tool: str) -> str:
    """Fill in the modules to check.

    A plain substitution, not str.format: the probe is Python source full of
    braces of its own, which format would try to read as fields.
    """

    return PROBE.replace(
        "__STRUCTBIO_MODULES__", repr(list(PROBE_MODULES.get(tool, ())))
    )


def parse_probe(output: str) -> ProbeResult:
    """Read the probe's line out of whatever conda printed around it."""

    for line in output.splitlines():
        if line.startswith("STRUCTBIO_PROBE "):
            try:
                values = json.loads(line[len("STRUCTBIO_PROBE ") :])
            except json.JSONDecodeError:
                continue
            return ProbeResult(ok=True, values=values)
    return ProbeResult(ok=False, error="the environment produced no usable answer")


def verify(tool: str, environment: str, timeout: float = 300.0) -> ProbeResult:
    """Run the probe inside a conda environment and report what it found."""

    try:
        completed = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", environment, "python", "-c",
             probe_source(tool)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return ProbeResult(ok=False, error="conda is not installed")
    except subprocess.SubprocessError as exc:
        return ProbeResult(ok=False, error=f"the check could not be run: {exc}")
    result = parse_probe(completed.stdout + "\n" + completed.stderr)
    if not result.ok and completed.returncode:
        tail = (completed.stderr or completed.stdout).strip().splitlines()
        result.error = tail[-1] if tail else f"exit code {completed.returncode}"
    return result


FACTS_PROBE = (
    "import json, sys, sysconfig; "
    "print('STRUCTBIO_FACTS ' + json.dumps({"
    "'python': '.'.join(str(p) for p in sys.version_info[:3]), "
    "'tag': 'cp' + str(sys.version_info[0]) + str(sys.version_info[1]), "
    "'platform': sysconfig.get_platform(), "
    "'executable': sys.executable}))"
)


def environment_python(name: str) -> Path | None:
    """The interpreter belonging to an environment, by path.

    Not `conda run -n NAME python`: that resolves through PATH, so an
    environment missing its own interpreter silently answers with the base
    one, and every version reported afterwards is about the wrong Python.
    """

    prefix = environment.conda_environments().get(name)
    if prefix is None:
        return None
    for relative in ("bin/python", "python.exe", "bin/python3"):
        candidate = prefix / relative
        if candidate.exists():
            return candidate
    return None


def environment_facts(name: str) -> dict[str, str]:
    """The Python version and platform inside an environment, for diagnostics.

    A "no matching distribution" from a wheel index is nearly always one of
    these two, and neither appears in pip's message.
    """

    interpreter = environment_python(name)
    if interpreter is None:
        prefix = environment.conda_environments().get(name)
        return {"missing_python": str(prefix) if prefix else "", "environment": name}
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", FACTS_PROBE],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    for line in (completed.stdout + completed.stderr).splitlines():
        if line.startswith("STRUCTBIO_FACTS "):
            try:
                return json.loads(line[len("STRUCTBIO_FACTS ") :])
            except json.JSONDecodeError:
                return {}
    return {}


def unusable_python(name: str) -> str | None:
    """Say so when an environment's Python has no wheels, before downloading any.

    Conda will happily build an environment on a Python that PyTorch does not
    publish for, and pip then reports "from versions: none" several minutes
    later without naming the reason.
    """

    facts = environment_facts(name)
    if "missing_python" in facts:
        where = facts["missing_python"] or "an unknown location"
        return (
            f"{name} exists at {where} but contains no Python interpreter, so the "
            "step that was supposed to create it did not finish. Nothing was "
            f"downloaded.\nRemove it with 'conda env remove -n {name}' and re-run; "
            "if it fails again, the conda output from the first step is what to send."
        )
    tag = facts.get("tag")
    if not tag or tag in SUPPORTED_PYTHON_TAGS:
        return None
    return (
        f"{name} has Python {facts.get('python', tag)} at "
        f"{facts.get('executable', 'an unknown path')}, which PyTorch and DGL "
        f"publish no CUDA wheels for; they cover "
        f"{', '.join(SUPPORTED_PYTHON_TAGS)}. Nothing was downloaded.\n"
        f"Remove it with 'conda env remove -n {name}' and re-run. If the path above "
        "is not inside that environment, conda answered with a different Python "
        "than the environment's own, which is worth reporting."
    )


def explain_pip_failure(name: str, output: str) -> list[str]:
    """Turn "No matching distribution" into the reason it actually happened."""

    if "No matching distribution" not in output and "from versions: none" not in output:
        return []
    facts = environment_facts(name)
    lines: list[str] = []
    if facts:
        lines.append(
            f"The environment {name!r} runs Python {facts.get('python', '?')} "
            f"({facts.get('tag', '?')}) on {facts.get('platform', '?')}."
        )
    tag = facts.get("tag", "")
    platform = facts.get("platform", "")
    supported_python = not tag or tag in SUPPORTED_PYTHON_TAGS
    supported_platform = not platform or "x86_64" in platform or "amd64" in platform

    if not supported_python:
        lines.append(
            f"PyTorch and DGL publish wheels for {', '.join(SUPPORTED_PYTHON_TAGS)}; "
            f"this environment is {tag}, which neither builds for."
        )
    if not supported_platform:
        lines.append(
            f"The platform is {platform}. The CUDA wheel indexes for PyTorch and "
            "DGL are x86_64 only, so an ARM machine cannot install from them. "
            "NVIDIA publishes its own PyTorch builds for ARM."
        )
    if supported_python and supported_platform:
        # The index does publish for this environment, so pip did not read it.
        lines.append(
            "This Python and platform ARE published for, so the index was not read "
            "properly rather than lacking the wheel. That is almost always the "
            "network: a proxy, a firewall, or an internal mirror."
        )
        lines.append("Check whether this machine can reach the index at all:")
        lines.append("  curl -sI https://download.pytorch.org/whl/cu118/torch/ | head -1")
        lines.append(
            "  echo $http_proxy $https_proxy $HTTP_PROXY $HTTPS_PROXY $no_proxy"
        )
        lines.append("  cat /etc/pip.conf ~/.pip/pip.conf ~/.config/pip/pip.conf")
        lines.append(
            "A site mirror configured in pip.conf, or a proxy that pip is not "
            "using, produces exactly this message. pip prints the underlying "
            "fetch failure only with -v."
        )
    return lines



def _pairing(needed: tuple[int, int], driver: tuple[int, int] | None) -> str | None:
    """The CUDA build that can drive this GPU and that this driver can load.

    The build must be at least what the card needs, or it holds no kernels for
    it, and at most what the driver supports, or it will not load at all.
    """

    for version, name in SUPPORTED_PAIRINGS:
        if version >= needed and (driver is None or version <= driver):
            return name
    return None


def machine_capability() -> tuple[int, int] | None:
    capabilities = environment.gpu_capabilities()
    return max(capabilities) if capabilities else None


def plan_environment(
    tool: str,
    installation: ToolInstallation,
    *,
    environment: str | None = None,
    capability: tuple[int, int] | None = None,
) -> EnvironmentPlan:
    """Work out how to build this tool's environment on this machine."""

    name = environment or installation.environment or _default_name(tool)
    if tool == "proteinmpnn":
        return _proteinmpnn_plan(name)
    if tool == "rfdiffusion":
        return _rfdiffusion_plan(name, installation, capability)
    return EnvironmentPlan(
        tool=tool,
        environment=name,
        blocked=(
            f"{tool} manages its own environment; structbio does not build one for it"
        ),
    )


def _default_name(tool: str) -> str:
    return {"rfdiffusion": "SE3nv", "proteinmpnn": "mlfold"}.get(tool, tool)


def _torch_build() -> str:
    return environment.select_torch_build(environment.driver_cuda_version())


def _proteinmpnn_plan(name: str) -> EnvironmentPlan:
    """ProteinMPNN needs only Python, PyTorch and NumPy, and pins nothing."""

    build = _torch_build()
    index = f"https://download.pytorch.org/whl/{build}"
    return EnvironmentPlan(
        tool="proteinmpnn",
        environment=name,
        steps=(
            Step(
                f"create the conda environment {name}",
                ("conda", "create", "-y", "-n", name, "python=3.11", "numpy"),
            ),
            Step(
                f"install PyTorch ({build})",
                ("conda", "run", "-n", name, "pip", "install", "torch",
                 "--index-url", index),
            ),
        ),
        notes=(
            "ProteinMPNN pins no versions, so the current PyTorch for this "
            "machine is the right one.",
            "Its model weights ship inside the repository; nothing else is downloaded.",
        ),
    )


def _rfdiffusion_plan(
    name: str, installation: ToolInstallation, capability: tuple[int, int] | None = None
) -> EnvironmentPlan:
    """RFdiffusion needs PyTorch, DGL and its bundled SE3Transformer to agree."""

    checkout = installation.path.expanduser() if installation.path else None
    capability = capability or machine_capability()

    if capability is None:
        # A card that is present but unidentified is a different problem from no
        # card at all, and saying "no GPU" to someone looking at one is useless.
        report = environment.gpu_report()
        if report.available:
            return EnvironmentPlan(
                tool="rfdiffusion",
                environment=name,
                blocked=(
                    f"this machine has {report.names[0]}, but its compute capability "
                    "could not be determined, and the right versions depend on it"
                ),
                alternatives=(
                    "state it yourself: structbio env create rfdiffusion --capability 8.9",
                    "find it with: nvidia-smi --query-gpu=compute_cap --format=csv",
                    "or look the card up in NVIDIA's CUDA GPUs table",
                ),
            )
        return EnvironmentPlan(
            tool="rfdiffusion",
            environment=name,
            blocked=(
                "no NVIDIA GPU was found, and RFdiffusion is not usable on a CPU "
                "in any practical sense"
                + (f". {report.error}" if report.error else "")
            ),
            alternatives=(
                "check the detection itself with: structbio gpu",
                "run it on a machine with an NVIDIA GPU",
            ),
        )
    if checkout is None:
        return EnvironmentPlan(
            tool="rfdiffusion",
            environment=name,
            blocked="no RFdiffusion checkout is configured",
            alternatives=("structbio install rfdiffusion --into ~/software",),
        )

    requirement = environment.required_cuda(capability)
    needed = requirement[0] if requirement else (11, 1)
    architecture = requirement[1] if requirement else "this GPU"

    # The upstream pins work on Ampere and older; the only thing wrong with
    # env/SE3nv.yml there is its channel order, which lets conda pick a
    # CPU-only PyTorch, so the channels are given explicitly instead.
    if needed <= (11, 1):
        return EnvironmentPlan(
            tool="rfdiffusion",
            environment=name,
            steps=(
                Step(
                    f"create {name} with the versions env/SE3nv.yml pins",
                    ("conda", "create", "-y", "-n", name,
                     "-c", "pytorch", "-c", "nvidia", "-c", "dglteam", "-c", "conda-forge",
                     "python=3.9", "pytorch=1.9", "torchvision", "torchaudio",
                     "cudatoolkit=11.1", "dgl-cuda11.1"),
                ),
                *_rfdiffusion_common(name, checkout),
            ),
            notes=(
                f"This GPU is {architecture}, which the pinned versions support.",
                "The channels are named in priority order, which env/SE3nv.yml does "
                "not do; that is why conda otherwise installs a CPU-only PyTorch.",
            ),
        )

    driver = environment.driver_cuda_version()
    build = _pairing(needed, driver)
    if build is None:
        newest = SUPPORTED_PAIRINGS[-1][0]
        return EnvironmentPlan(
            tool="rfdiffusion",
            environment=name,
            blocked=(
                f"this GPU is {architecture}, needing CUDA {needed[0]}.{needed[1]}, "
                f"but PyTorch {TORCH_RELEASE} and DGL are only published together up "
                f"to CUDA {newest[0]}.{newest[1]}. RFdiffusion needs DGL, so there is "
                "no version combination that runs it on this card"
            ),
            alternatives=(
                "run RFdiffusion on an older card (Ampere, Ada or Hopper)",
                "ask the RFdiffusion maintainers whether a newer DGL is supported yet",
                "the other wrapped tools are unaffected: ProteinMPNN, ColabFold and "
                "CryoZeta do not use DGL",
            ),
        )

    torch_index = f"https://download.pytorch.org/whl/{build}"
    dgl_index = DGL_INDEX.format(torch=DGL_TORCH_INDEX, build=build)
    return EnvironmentPlan(
        tool="rfdiffusion",
        environment=name,
        upstream_verified=False,
        steps=(
            Step(
                f"create the conda environment {name}",
                ("conda", "create", "-y", "-n", name, "python=3.10"),
            ),
            Step(
                f"check that PyTorch {TORCH_RELEASE} ({build}) can be installed here",
                ("conda", "run", "-n", name, "pip", "install", "--dry-run",
                 f"torch=={TORCH_RELEASE}", "--index-url", torch_index),
            ),
            Step(
                f"install PyTorch {TORCH_RELEASE} ({build})",
                ("conda", "run", "-n", name, "pip", "install",
                 f"torch=={TORCH_RELEASE}", "torchvision", "torchaudio",
                 "--index-url", torch_index),
            ),
            Step(
                f"install DGL for PyTorch {DGL_TORCH_INDEX} ({build})",
                ("conda", "run", "-n", name, "pip", "install", "dgl", "-f", dgl_index),
            ),
            *_rfdiffusion_common(name, checkout),
        ),
        notes=(
            f"This GPU is {architecture}, which needs CUDA "
            f"{needed[0]}.{needed[1]} or newer, so the versions env/SE3nv.yml pins "
            "cannot drive it.",
            f"PyTorch {TORCH_RELEASE} with DGL {build} is a pairing both projects "
            "publish for the same CUDA.",
            "This combination is NOT one the RFdiffusion authors have published. "
            "It is checked by running code on the GPU, but check your first "
            "designs against a known result before trusting a campaign to it.",
        ),
    )


def _rfdiffusion_common(name: str, checkout: Path) -> tuple[Step, ...]:
    return (
        Step(
            "install the SE3Transformer requirements",
            ("conda", "run", "-n", name, "pip", "install", *SE3_REQUIREMENTS),
        ),
        Step(
            "install NVIDIA's dllogger, which SE3Transformer imports",
            ("conda", "run", "-n", name, "pip", "install",
             "git+https://github.com/NVIDIA/dllogger#egg=dllogger"),
        ),
        Step(
            "install the RFdiffusion requirements",
            ("conda", "run", "-n", name, "pip", "install", *RFDIFFUSION_EXTRAS),
        ),
        Step(
            "install the bundled SE3Transformer",
            ("conda", "run", "-n", name, "pip", "install", "--no-deps", "-e",
             str(checkout / "env" / "SE3Transformer")),
        ),
        Step(
            "install RFdiffusion itself",
            ("conda", "run", "-n", name, "pip", "install", "--no-deps", "-e", str(checkout)),
        ),
    )


def backup_name(name: str) -> str:
    """A free name to move an environment aside to, never overwriting one."""

    existing = environment.conda_environments()
    for index in range(1, 100):
        candidate = f"{name}-before-{index}"
        if candidate not in existing:
            return candidate
    raise RuntimeError(f"No free backup name for {name}")


def move_aside(name: str) -> str | None:
    """Rename an environment out of the way. Returns the new name, or None.

    Deleting is not structbio's to do: an environment a researcher built may be
    the only working one on the machine, and rebuilding it is hours of their
    time even when it succeeds.
    """

    target = backup_name(name)
    try:
        completed = subprocess.run(
            ["conda", "rename", "-n", name, target],
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode:
        return None
    return target


def environment_exists(name: str) -> bool:
    return name in environment.conda_environments()
