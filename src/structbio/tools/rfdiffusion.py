"""RFdiffusion backend using its documented Hydra CLI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from structbio.config import ResourceConfig, ToolInstallation, resolve_from_config
from structbio.tools.base import (
    BackendContext,
    CommandPlan,
    CommandStep,
    EnvironmentCheck,
    ToolBackend,
    ValidationReport,
    standard_environment_check,
    wrap_environment,
)
from structbio.validation import (
    StructureValidationError,
    parse_pdb,
    parse_position_spec,
    validate_contig,
)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class RFInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pdb: Path | None = None


class RFDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "monomer", "symmetry", "motif", "binder", "partial", "inpainting"
    ] = "monomer"
    symmetry: str | None = None
    length: int | None = Field(default=None, ge=1, le=10_000)
    num_designs: int = Field(default=1, ge=1, le=1_000_000)
    contigs: list[str] = Field(default_factory=list)
    hotspot_residues: list[str] = Field(default_factory=list)
    inpaint_sequence: list[str] = Field(default_factory=list)
    inpaint_structure: list[str] = Field(default_factory=list)


class RFDiffusionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timesteps: int | None = Field(default=None, ge=1, le=1_000)
    partial_t: int | None = Field(default=None, ge=1, le=1_000)
    guide_scale: float | None = Field(default=None, ge=0, le=100)
    guide_decay: Literal["constant", "linear", "quadratic", "cubic"] | None = None


class RFDiffusionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["rfdiffusion"]
    experiment: ExperimentSpec
    input: RFInput = Field(default_factory=RFInput)
    design: RFDesign = Field(default_factory=RFDesign)
    potentials: dict[str, dict[str, str | int | float | bool]] = Field(default_factory=dict)
    diffusion: RFDiffusionSettings = Field(default_factory=RFDiffusionSettings)
    resources: ResourceConfig = Field(default_factory=lambda: ResourceConfig(gpus=1))
    hydra_overrides: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_mode_fields(self) -> "RFDiffusionConfig":
        if self.design.mode == "symmetry" and not self.design.symmetry:
            raise ValueError("design.symmetry is required in symmetry mode")
        if self.design.mode in {"motif", "binder", "partial", "inpainting"} and not self.input.pdb:
            raise ValueError(f"input.pdb is required in {self.design.mode} mode")
        if self.design.mode == "partial" and self.diffusion.partial_t is None:
            raise ValueError("diffusion.partial_t is required in partial mode")
        if not self.design.contigs and self.design.length is None:
            raise ValueError("design.length or design.contigs must be provided")
        return self


_HYDRA_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_SYMMETRY_RE = re.compile(r"^(?:c[2-9]\d*|d[2-9]\d*|tetrahedral)$", re.IGNORECASE)
_PDB_POSITION_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])(-?\d+)(?:-(-?\d+))?")


def _hydra_list(items: list[str]) -> str:
    return "[" + ",".join(items) + "]"


class RFDiffusionBackend(ToolBackend):
    name = "rfdiffusion"
    display_name = "RFdiffusion"
    config_model = RFDiffusionConfig

    def parse_config(self, raw: dict[str, Any], source: Path) -> RFDiffusionConfig:
        selected = {
            key: raw[key]
            for key in (
                "tool",
                "experiment",
                "input",
                "design",
                "potentials",
                "diffusion",
                "resources",
                "hydra_overrides",
            )
            if key in raw
        }
        config = RFDiffusionConfig.model_validate(selected)
        if config.input.pdb:
            config.input.pdb = resolve_from_config(config.input.pdb, source)
        return config

    def validate(self, config: RFDiffusionConfig) -> ValidationReport:
        report = ValidationReport()
        structure = None
        if config.input.pdb:
            try:
                structure = parse_pdb(config.input.pdb)
            except StructureValidationError as exc:
                report.error(str(exc))
        for contig in self._contigs(config):
            try:
                validate_contig(contig)
            except StructureValidationError as exc:
                report.error(str(exc))
        if config.design.symmetry and not _SYMMETRY_RE.fullmatch(config.design.symmetry):
            report.error(
                "Unsupported symmetry; use cN, dN, or tetrahedral as documented by RFdiffusion"
            )
        if config.diffusion.partial_t and config.diffusion.timesteps:
            if config.diffusion.partial_t > config.diffusion.timesteps:
                report.error("diffusion.partial_t cannot exceed diffusion.timesteps")
        if structure:
            references: list[str] = []
            for contig in config.design.contigs:
                references.extend(
                    f"{match.group(1)}:{match.group(2)}-{match.group(3)}"
                    if match.group(3)
                    else f"{match.group(1)}:{match.group(2)}"
                    for match in _PDB_POSITION_RE.finditer(contig)
                )
            references.extend(_colon_position(item) for item in config.design.hotspot_residues)
            references.extend(_colon_position(item) for item in config.design.inpaint_sequence)
            references.extend(_colon_position(item) for item in config.design.inpaint_structure)
            for reference in references:
                try:
                    parse_position_spec(reference, structure)
                except StructureValidationError as exc:
                    report.error(str(exc))
        for key in config.hydra_overrides:
            if not _HYDRA_KEY_RE.fullmatch(key):
                report.error(f"Unsafe or malformed Hydra override key: {key!r}")
        report.details.append(f"Mode: {config.design.mode}")
        report.details.append(f"Designs: {config.design.num_designs}")
        return report

    def _contigs(self, config: RFDiffusionConfig) -> list[str]:
        if config.design.contigs:
            return config.design.contigs
        assert config.design.length is not None
        if config.design.mode == "symmetry":
            return [str(config.design.length)]
        return [f"{config.design.length}-{config.design.length}"]

    def build_command(self, config: RFDiffusionConfig, context: BackendContext) -> CommandPlan:
        executable = context.installation.executable or "scripts/run_inference.py"
        script = (
            str((context.installation.path / executable).resolve())
            if context.installation.path
            else executable
        )
        argv = ["python", script]
        if config.design.mode == "symmetry":
            argv.extend(["--config-name", "symmetry"])
            argv.append(f"inference.symmetry={config.design.symmetry}")
        argv.extend(
            [
                f"contigmap.contigs={_hydra_list(self._contigs(config))}",
                f"inference.output_prefix={context.output_dir / config.experiment.name}",
                f"inference.num_designs={config.design.num_designs}",
            ]
        )
        if config.input.pdb:
            argv.append(f"inference.input_pdb={config.input.pdb}")
        if config.design.hotspot_residues:
            argv.append(f"ppi.hotspot_res={_hydra_list(config.design.hotspot_residues)}")
        if config.design.inpaint_sequence:
            argv.append(
                f"contigmap.inpaint_seq={_hydra_list(config.design.inpaint_sequence)}"
            )
        if config.design.inpaint_structure:
            argv.append(
                f"contigmap.inpaint_str={_hydra_list(config.design.inpaint_structure)}"
            )
        if config.diffusion.timesteps is not None:
            argv.append(f"diffuser.T={config.diffusion.timesteps}")
        if config.diffusion.partial_t is not None:
            argv.append(f"diffuser.partial_T={config.diffusion.partial_t}")
        if config.potentials:
            potentials = []
            for name, parameters in config.potentials.items():
                entries = [f"type:{name}"] + [
                    f"{key}:{_scalar(value)}" for key, value in parameters.items()
                ]
                potentials.append('"' + ",".join(entries) + '"')
            argv.append(f"potentials.guiding_potentials={_hydra_list(potentials)}")
        if config.diffusion.guide_scale is not None:
            argv.append(f"potentials.guide_scale={config.diffusion.guide_scale}")
        if config.diffusion.guide_decay is not None:
            argv.append(f"potentials.guide_decay={config.diffusion.guide_decay}")
        argv.extend(f"{key}={_scalar(value)}" for key, value in config.hydra_overrides.items())
        inputs = [str(config.input.pdb)] if config.input.pdb else []
        return CommandPlan(
            steps=[
                CommandStep(
                    wrap_environment(argv, context.installation),
                    cwd=context.installation.path,
                )
            ],
            output_dir=context.output_dir,
            artifacts={"absolute_input_paths": inputs},
        )

    def check_environment(self, installation: ToolInstallation) -> EnvironmentCheck:
        return standard_environment_check(
            installation,
            tool=self.name,
            default_executable="scripts/run_inference.py",
        )


def _scalar(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _colon_position(value: str) -> str:
    value = value.strip()
    if ":" in value:
        return value
    match = re.fullmatch(r"([A-Za-z])(-?\d+(?:-\d+)?)", value)
    return f"{match.group(1)}:{match.group(2)}" if match else value
