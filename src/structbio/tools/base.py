"""Backend interface shared by all scientific tools."""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from structbio.config import ToolInstallation


@dataclass(frozen=True)
class ValidationMessage:
    level: str
    message: str


@dataclass
class ValidationReport:
    messages: list[ValidationMessage] = field(default_factory=list)
    details: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [item.message for item in self.messages if item.level == "error"]

    @property
    def warnings(self) -> list[str]:
        return [item.message for item in self.messages if item.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.messages.append(ValidationMessage("error", message))

    def warning(self, message: str) -> None:
        self.messages.append(ValidationMessage("warning", message))


@dataclass(frozen=True)
class CommandStep:
    argv: tuple[str, ...]
    name: str = "main"
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        prefix = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(self.env.items())
        )
        command = shlex.join(self.argv)
        return f"{prefix} {command}" if prefix else command


@dataclass
class CommandPlan:
    steps: list[CommandStep]
    output_dir: Path
    artifacts: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return "\n".join(step.render() for step in self.steps)


@dataclass(frozen=True)
class BackendContext:
    source: Path
    installation: ToolInstallation
    experiment_dir: Path
    output_dir: Path
    inputs_dir: Path


@dataclass(frozen=True)
class EnvironmentCheck:
    configured: bool
    found: bool
    executable: str | None = None
    details: tuple[str, ...] = ()


class ToolBackend(ABC):
    name: str
    display_name: str
    config_model: type[BaseModel]

    @abstractmethod
    def parse_config(self, raw: dict[str, Any], source: Path) -> BaseModel:
        raise NotImplementedError

    @abstractmethod
    def validate(self, config: BaseModel) -> ValidationReport:
        raise NotImplementedError

    @abstractmethod
    def build_command(self, config: BaseModel, context: BackendContext) -> CommandPlan:
        raise NotImplementedError

    @abstractmethod
    def check_environment(self, installation: ToolInstallation) -> EnvironmentCheck:
        raise NotImplementedError

    def materialize_artifacts(self, plan: CommandPlan) -> None:
        """Write generated files only inside a newly created experiment."""

    def collect_outputs(self, experiment_dir: Path) -> list[Path]:
        output_dir = experiment_dir / "outputs"
        return sorted(path for path in output_dir.rglob("*") if path.is_file())


def wrap_environment(argv: list[str], installation: ToolInstallation) -> tuple[str, ...]:
    """Wrap a command without relying on interactive shell activation."""

    if installation.manager == "conda" and installation.environment:
        return (
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            installation.environment,
            *argv,
        )
    return tuple(argv)
