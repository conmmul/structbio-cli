"""Answer one question about each tool: can it run right now?

`structbio setup` uses this to finish the job it starts. Finding a checkout on
disk is not the same as being able to run it — almost every setup problem in
this lab is an environment whose PyTorch cannot use the graphics card — so
setup runs the same check `structbio env verify` runs, and says which single
command fixes what it finds.

Checking runs code; it never installs, removes or rebuilds anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from structbio import discovery, environment, provision
from structbio.config import StructbioSettings, ToolInstallation, user_config_path
from structbio.tools import get_backends


READY = "ready"
BROKEN = "needs attention"
NO_ENVIRONMENT = "no environment"
UNCHECKED = "not checked"


@dataclass(frozen=True)
class ToolStatus:
    """What one tool can do on this machine, and the command that fixes it."""

    tool: str
    state: str
    environment: str | None = None
    detail: str = ""
    fix: str | None = None
    adopted: bool = False

    @property
    def ready(self) -> bool:
        return self.state == READY


def _candidate_environments(tool: str) -> tuple[str, ...]:
    for signature in discovery.SIGNATURES:
        if signature.tool == tool:
            return signature.environment_names
    return ()


def _probe(tool: str, installation: ToolInstallation, name: str) -> provision.ProbeResult | None:
    """Run the check inside an environment, or None when there is no such environment."""

    interpreter = provision.tool_interpreter(installation, name)
    if interpreter is None:
        return None
    return provision.verify(tool, name, interpreter=interpreter)


def _create_hint(tool: str, installation: ToolInstallation) -> str:
    if installation.manager == "pixi":
        return f"cd {installation.path} && pixi run setup"
    return f"structbio env create {tool}"


def _repair_hint(tool: str, installation: ToolInstallation) -> str:
    if installation.manager == "pixi":
        return f"cd {installation.path} && pixi run setup"
    return f"structbio env repair {tool}"


def check(tool: str, installation: ToolInstallation) -> ToolStatus:
    """Check one configured tool, adopting a working environment if it has none."""

    backend = get_backends().get(tool)
    if backend is None or not backend.needs_torch:
        return ToolStatus(tool, UNCHECKED, detail="manages its own environment")

    name = installation.environment
    if name:
        result = _probe(tool, installation, name)
        if result is None:
            return ToolStatus(
                tool,
                NO_ENVIRONMENT,
                name,
                f"the configured environment {name!r} does not exist",
                _create_hint(tool, installation),
            )
        if not result.failures():
            return ToolStatus(tool, READY, name, result.summary())
        return ToolStatus(tool, BROKEN, name, result.failures()[0], _repair_hint(tool, installation))

    # No environment is recorded: an environment that already works is the
    # right answer, and far better than building a second one beside it.
    existing = environment.conda_environments()
    for candidate in _candidate_environments(tool):
        if candidate not in existing:
            continue
        result = _probe(tool, installation, candidate)
        if result is not None and not result.failures():
            return ToolStatus(tool, READY, candidate, result.summary(), adopted=True)
    return ToolStatus(
        tool,
        NO_ENVIRONMENT,
        detail="no working environment was found",
        fix=_create_hint(tool, installation),
    )


def review(settings: StructbioSettings) -> list[ToolStatus]:
    """Check every configured tool, in the order the backends are registered."""

    return [
        check(tool, installation)
        for tool, installation in ((name, settings.tools.get(name)) for name in get_backends())
        if installation is not None
    ]


def record_environment(tool: str, name: str, config_path: Path | None = None) -> Path | None:
    """Write an environment that passed its check into the user configuration."""

    path = config_path or user_config_path()
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        data = yaml.safe_load(existing) or {}
        if not isinstance(data, dict):
            return None
        entry = data.setdefault("tools", {}).setdefault(tool, {})
        if entry.get("environment") == name:
            return None
        entry["environment"] = name
        entry.setdefault("manager", "conda")
        if existing:
            path.with_suffix(path.suffix + ".bak").write_text(existing, encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    except (OSError, ValueError, yaml.YAMLError):
        return None
    return path
