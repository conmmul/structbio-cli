"""Find scientific software that is already installed on this machine.

Nothing here downloads or changes anything: it looks in the places these tools
are normally put, and reports what it finds so the configuration can be written
without the researcher hunting for paths.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from structbio.environment import command_output


# Directories to look in, in order. Deeper trees are not searched: a checkout
# buried five levels down is rarer than a home directory large enough to make
# scanning slow.
SEARCH_ROOTS: tuple[str, ...] = (
    "~/software",
    "~/apps",
    "~/src",
    "~/opt",
    "~/tools",
    "~",
    "/opt",
    "/usr/local",
    "/usr/local/share",
)
SEARCH_DEPTH = 2
MAX_ENTRIES = 4000


@dataclass(frozen=True)
class ToolSignature:
    """How one wrapped tool can be recognised on disk."""

    tool: str
    directory_names: tuple[str, ...]
    marker: str
    manager: str
    environment_names: tuple[str, ...] = ()
    command: str | None = None
    # Relative paths to check inside a directory found by name, for tools that
    # install their executable into a nested environment.
    nested_commands: tuple[str, ...] = ()


SIGNATURES: tuple[ToolSignature, ...] = (
    ToolSignature(
        tool="rfdiffusion",
        directory_names=("RFdiffusion", "rfdiffusion"),
        marker="scripts/run_inference.py",
        manager="conda",
        environment_names=("SE3nv", "rfdiffusion", "SE3nv_ampere"),
    ),
    ToolSignature(
        tool="proteinmpnn",
        directory_names=("ProteinMPNN", "proteinmpnn"),
        marker="protein_mpnn_run.py",
        manager="conda",
        # Only names specific to this tool: borrowing another tool's environment
        # would put a wrong 'conda run -n' in front of every command.
        environment_names=("mlfold", "proteinmpnn"),
    ),
    ToolSignature(
        tool="colabfold",
        directory_names=("localcolabfold", "colabfold", "ColabFold"),
        marker="",
        manager="none",
        environment_names=("colabfold", "localcolabfold"),
        command="colabfold_batch",
        nested_commands=(
            ".pixi/envs/default/bin/colabfold_batch",
            "colabfold-conda/bin/colabfold_batch",
            "bin/colabfold_batch",
        ),
    ),
    ToolSignature(
        tool="cryozeta",
        directory_names=("CryoZeta", "cryozeta"),
        marker="inference_demo.sh",
        manager="pixi",
        environment_names=("default",),
    ),
)


@dataclass(frozen=True)
class Discovery:
    """One installed tool, and how it was found."""

    tool: str
    path: Path | None
    executable: str
    manager: str
    environment: str | None
    found_by: str

    def settings(self) -> dict[str, object]:
        entry: dict[str, object] = {"executable": self.executable, "manager": self.manager}
        if self.path is not None:
            entry["path"] = str(self.path)
        if self.environment:
            entry["environment"] = self.environment
        return entry

    def describe(self) -> str:
        where = str(self.path) if self.path else self.executable
        suffix = f" (environment: {self.environment})" if self.environment else ""
        return f"{where}{suffix}"


def conda_environments() -> dict[str, Path]:
    """Map conda environment names to their prefixes, if conda is installed."""

    if shutil.which("conda") is None and shutil.which("mamba") is None:
        return {}
    manager = "conda" if shutil.which("conda") else "mamba"
    output = command_output([manager, "env", "list", "--json"], timeout=20.0, only_on_success=True)
    if not output:
        return {}
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return {}
    environments: dict[str, Path] = {}
    for raw in payload.get("envs", []):
        prefix = Path(raw)
        if prefix.is_dir():
            environments[prefix.name] = prefix
    return environments


def candidate_directories(roots: tuple[str, ...] = SEARCH_ROOTS) -> list[Path]:
    """List directories worth inspecting, breadth first and bounded."""

    seen: set[Path] = set()
    found: list[Path] = []
    budget = MAX_ENTRIES
    for raw in roots:
        root = Path(raw).expanduser()
        if not root.is_dir():
            continue
        level = [root]
        for _ in range(SEARCH_DEPTH):
            children: list[Path] = []
            for directory in level:
                if budget <= 0:
                    return found
                try:
                    entries = sorted(directory.iterdir())
                except (OSError, PermissionError):
                    continue
                for entry in entries:
                    budget -= 1
                    if budget <= 0:
                        return found
                    if not entry.is_dir() or entry.name.startswith("."):
                        continue
                    resolved = entry.resolve()
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    found.append(entry)
                    children.append(entry)
            level = children
    return found


def _environment_for(
    signature: ToolSignature, environments: dict[str, Path]
) -> str | None:
    for name in signature.environment_names:
        if name in environments:
            return name
    return None


def _from_command(signature: ToolSignature, environments: dict[str, Path]) -> Discovery | None:
    if not signature.command:
        return None
    on_path = shutil.which(signature.command)
    if on_path:
        return Discovery(
            tool=signature.tool,
            path=None,
            executable=signature.command,
            manager="none",
            environment=None,
            found_by="PATH",
        )
    for name, prefix in environments.items():
        candidate = prefix / "bin" / signature.command
        if candidate.is_file():
            return Discovery(
                tool=signature.tool,
                path=None,
                executable=str(candidate),
                manager="none",
                environment=None,
                found_by=f"conda environment {name}",
            )
    return None


def _from_directory(
    signature: ToolSignature, directories: list[Path], environments: dict[str, Path]
) -> Discovery | None:
    wanted = {name.lower() for name in signature.directory_names}
    for directory in directories:
        if directory.name.lower() not in wanted:
            continue
        if signature.marker and (directory / signature.marker).is_file():
            return Discovery(
                tool=signature.tool,
                path=directory,
                executable=signature.marker,
                manager=signature.manager,
                environment=_environment_for(signature, environments),
                found_by=str(directory.parent),
            )
        for relative in signature.nested_commands:
            if (directory / relative).is_file():
                return Discovery(
                    tool=signature.tool,
                    path=None,
                    executable=str(directory / relative),
                    manager="none",
                    environment=None,
                    found_by=str(directory),
                )
    return None


def discover(
    signatures: tuple[ToolSignature, ...] = SIGNATURES,
    *,
    roots: tuple[str, ...] = SEARCH_ROOTS,
) -> dict[str, Discovery]:
    """Look for every known tool, most reliable evidence first."""

    environments = conda_environments()
    directories = candidate_directories(roots)
    results: dict[str, Discovery] = {}
    for signature in signatures:
        found = _from_command(signature, environments) or _from_directory(
            signature, directories, environments
        )
        if found is not None:
            results[signature.tool] = found
    return results


def render_config(
    discoveries: dict[str, Discovery],
    signatures: tuple[ToolSignature, ...] = SIGNATURES,
    experiments_root: str = "~/structbio-experiments",
) -> str:
    """Write a configuration naming what was found, and what was not."""

    lines = [
        "# structbio workstation configuration.",
        "#",
        "# Written by 'structbio setup' from the software found on this machine.",
        "# `path` is the root of a tool checkout and `executable` is relative to",
        "# it; a tool already on PATH needs no path. structbio never installs the",
        "# scientific software or its model weights.",
        "",
        # An empty mapping, so the file stays valid when nothing was found: a
        # bare "tools:" above comments alone parses as null and breaks loading.
        "tools: {}" if not discoveries else "tools:",
    ]
    for signature in signatures:
        found = discoveries.get(signature.tool)
        if found is None:
            lines.append(f"  # {signature.tool}: not found on this machine.")
            lines.append(f"  #   Install it with: structbio install {signature.tool}")
            continue
        lines.append(f"  {signature.tool}:")
        for key, value in found.settings().items():
            lines.append(f"    {key}: {value}")
        if not found.environment and signature.environment_names:
            lines.append(
                f"    # environment: {signature.environment_names[0]}"
                "   # set this if the tool needs one"
            )
    lines.extend(
        [
            "",
            "# Only used by 'structbio TOOL run CONFIG.yaml'. Quick commands write to",
            "# the output folder named on the command line instead.",
            f"experiments_root: {experiments_root}",
            "",
        ]
    )
    return "\n".join(lines)


def merge_into_config(existing: str, discoveries: dict[str, Discovery]) -> str:
    """Add newly found tools to a configuration, leaving existing entries alone."""

    import yaml

    data = yaml.safe_load(existing) or {}
    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")
    tools = data.get("tools") or {}
    if not isinstance(tools, dict):
        raise ValueError("'tools' must be a mapping")
    data["tools"] = tools
    added: list[str] = []
    for name, found in discoveries.items():
        if name in tools:
            continue
        tools[name] = found.settings()
        added.append(name)
    if not added:
        return existing
    header = "# Updated by 'structbio setup': added " + ", ".join(sorted(added)) + ".\n"
    return header + yaml.safe_dump(data, sort_keys=False)


def environment_summary() -> dict[str, str]:
    """A short description of the package managers available here."""

    return {
        "conda": shutil.which("conda") or shutil.which("mamba") or "",
        "pixi": shutil.which("pixi") or "",
        "git": shutil.which("git") or "",
        "PATH entries": str(len(os.environ.get("PATH", "").split(os.pathsep))),
    }
