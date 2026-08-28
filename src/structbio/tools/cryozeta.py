"""Thin adapter for the official CryoZeta inference_demo.sh interface."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from structbio.config import ResourceConfig, ToolInstallation, resolve_from_config
from structbio.environment import executable_path
from structbio.tools.base import (
    BackendContext,
    CommandPlan,
    CommandStep,
    EnvironmentCheck,
    ToolBackend,
    ValidationReport,
)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class CryoInput(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    json_path: Path = Field(alias="json")


class CryoZetaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["cryozeta"]
    experiment: ExperimentSpec
    input: CryoInput
    mode: Literal["combined", "cryozeta", "cryozeta-interpolate"] = "combined"
    pixi_environment: str | None = None
    gpu_ids: list[int] = Field(default_factory=list)
    checkpoint: Path | None = None
    interpolation_checkpoint: Path | None = None
    resources: ResourceConfig = Field(default_factory=lambda: ResourceConfig(gpus=1))

    @model_validator(mode="after")
    def unique_gpu_ids(self) -> "CryoZetaConfig":
        if any(value < 0 for value in self.gpu_ids):
            raise ValueError("gpu_ids cannot contain negative values")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids cannot contain duplicates")
        return self


class CryoZetaBackend(ToolBackend):
    """Adapter verified against kiharalab/CryoZeta main as of 2026-08-27."""

    name = "cryozeta"
    display_name = "CryoZeta"
    config_model = CryoZetaConfig

    def parse_config(self, raw: dict[str, Any], source: Path) -> CryoZetaConfig:
        selected = {
            key: raw[key]
            for key in (
                "tool",
                "experiment",
                "input",
                "mode",
                "pixi_environment",
                "gpu_ids",
                "checkpoint",
                "interpolation_checkpoint",
                "resources",
            )
            if key in raw
        }
        config = CryoZetaConfig.model_validate(selected)
        config.input.json_path = resolve_from_config(config.input.json_path, source)
        if config.checkpoint:
            config.checkpoint = resolve_from_config(config.checkpoint, source)
        if config.interpolation_checkpoint:
            config.interpolation_checkpoint = resolve_from_config(
                config.interpolation_checkpoint, source
            )
        return config

    def validate(self, config: CryoZetaConfig) -> ValidationReport:
        report = ValidationReport()
        if not config.input.json_path.is_file():
            report.error(f"CryoZeta input JSON does not exist: {config.input.json_path}")
            return report
        try:
            payload = json.loads(config.input.json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.error(f"Invalid CryoZeta input JSON: {exc}")
            return report
        if not isinstance(payload, list) or not payload:
            report.error("CryoZeta input JSON must be a non-empty list")
            return report
        required = {"name", "modelSeeds", "map_path", "resolution", "contour_level", "sequences"}
        for index, entry in enumerate(payload):
            if not isinstance(entry, dict):
                report.error(f"CryoZeta entry {index} is not an object")
                continue
            missing = required - set(entry)
            if missing:
                report.error(
                    f"CryoZeta entry {index} is missing: {', '.join(sorted(missing))}"
                )
            map_value = entry.get("map_path")
            if isinstance(map_value, str):
                map_path = Path(map_value).expanduser()
                if not map_path.is_absolute():
                    map_path = config.input.json_path.parent / map_path
                if not map_path.is_file():
                    report.error(f"CryoZeta map does not exist: {map_path.resolve()}")
            if not isinstance(entry.get("sequences"), list) or not entry.get("sequences"):
                report.error(f"CryoZeta entry {index} must contain at least one sequence")
        if config.resources.gpus > 1:
            report.warning(
                "The verified standard CryoZeta inference script runs each inference stage on one GPU"
            )
        report.details.append(f"Targets: {len(payload)}")
        report.details.append(f"Mode: {config.mode}")
        return report

    def build_command(self, config: CryoZetaConfig, context: BackendContext) -> CommandPlan:
        executable = context.installation.executable or "inference_demo.sh"
        script = (
            str((context.installation.path / executable).resolve())
            if context.installation.path
            else executable
        )
        argv = [
            "bash",
            script,
            "--input-json",
            str(config.input.json_path),
            "--output-dir",
            str(context.output_dir),
            "--mode",
            config.mode,
        ]
        pixi_environment = config.pixi_environment or context.installation.environment
        if pixi_environment:
            argv.extend(["--env", pixi_environment])
        if config.gpu_ids:
            argv.extend(["--gpu", ",".join(str(value) for value in config.gpu_ids)])
        if config.checkpoint:
            argv.extend(["--checkpoint", str(config.checkpoint)])
        if config.interpolation_checkpoint:
            argv.extend(["--interp-checkpoint", str(config.interpolation_checkpoint)])
        inputs = [config.input.json_path]
        if config.checkpoint:
            inputs.append(config.checkpoint)
        if config.interpolation_checkpoint:
            inputs.append(config.interpolation_checkpoint)
        return CommandPlan(
            steps=[
                CommandStep(
                    argv=tuple(argv),
                    cwd=context.installation.path,
                )
            ],
            output_dir=context.output_dir,
            artifacts={"absolute_input_paths": [str(path) for path in inputs]},
        )

    def check_environment(self, installation: ToolInstallation) -> EnvironmentCheck:
        executable = executable_path(installation)
        pixi = shutil.which("pixi")
        details = ["interface=inference_demo.sh (official repository)"]
        if installation.environment:
            details.append(f"environment={installation.environment}")
        if not pixi:
            details.append("pixi executable not found")
        return EnvironmentCheck(
            configured=installation.path is not None or executable is not None,
            found=bool(executable and pixi),
            executable=str(executable) if executable else None,
            details=tuple(details),
        )
