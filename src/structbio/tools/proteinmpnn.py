"""ProteinMPNN backend with explicit, inversion-checked mutation masks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
    ResidueId,
    StructureIndex,
    StructureValidationError,
    parse_pdb,
    parse_position_spec,
)


_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class MPNNInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pdb: Path | None = None
    directory: Path | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> "MPNNInput":
        if (self.pdb is None) == (self.directory is None):
            raise ValueError("Set exactly one of input.pdb or input.directory")
        return self


class MPNNDesign(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chains: list[str] = Field(default_factory=list)
    designable_positions: list[str] = Field(default_factory=list)
    fixed_positions: list[str] = Field(default_factory=list)
    num_sequences: int = Field(default=1, ge=1, le=1_000_000)
    temperatures: list[float] = Field(default_factory=lambda: [0.1])
    batch_size: int = Field(default=1, ge=1)
    seed: int = Field(default=0, ge=0)
    soluble_model: bool = False
    model_name: str = "v_48_020"

    @model_validator(mode="after")
    def validate_sampling(self) -> "MPNNDesign":
        if any(value <= 0 or value > 1 for value in self.temperatures):
            raise ValueError("design.temperatures must be greater than 0 and at most 1")
        if self.num_sequences % self.batch_size:
            raise ValueError("design.num_sequences must be divisible by design.batch_size")
        if any(len(chain) != 1 for chain in self.chains):
            raise ValueError("ProteinMPNN chain identifiers must be one character")
        return self


class MPNNConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    omit_aas: str = "X"
    omit_by_position: dict[str, str] = Field(default_factory=dict)
    bias_aas: dict[str, float] = Field(default_factory=dict)
    bias_by_position: dict[str, dict[str, float]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_amino_acids(self) -> "MPNNConstraints":
        values = [self.omit_aas, *self.omit_by_position.values()]
        keys = list(self.bias_aas) + [
            aa for values_by_aa in self.bias_by_position.values() for aa in values_by_aa
        ]
        invalid = set("".join(values) + "".join(keys)) - _ALPHABET
        if invalid:
            raise ValueError(f"Unknown amino-acid codes: {''.join(sorted(invalid))}")
        return self


class ProteinMPNNConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["proteinmpnn"]
    experiment: ExperimentSpec
    input: MPNNInput
    design: MPNNDesign = Field(default_factory=MPNNDesign)
    constraints: MPNNConstraints = Field(default_factory=MPNNConstraints)
    resources: ResourceConfig = Field(default_factory=lambda: ResourceConfig(gpus=1))


@dataclass(frozen=True)
class MaskInspection:
    pdb: Path
    structure: StructureIndex
    design_chains: tuple[str, ...]
    designable: frozenset[ResidueId]
    fixed: frozenset[ResidueId]
    fixed_dictionary: dict[str, dict[str, list[int]]]

    def render(self) -> str:
        designable = _compact_labels(self.designable)
        fixed = _compact_labels(self.fixed)
        lines = [
            "ProteinMPNN validation",
            "",
            f"Structure: {self.pdb.name}",
            "",
            f"Design chains: {' '.join(self.design_chains)}",
            "",
            "Designable residues:",
            designable or "(none)",
            "",
            f"Designable count: {len(self.designable)}",
            "",
            "Fixed residues:",
            fixed or "(none)",
            "",
            f"Fixed count: {len(self.fixed)}",
        ]
        return "\n".join(lines)


def inspect_mask(config: ProteinMPNNConfig, pdb: Path) -> MaskInspection:
    structure = parse_pdb(pdb)
    chains = tuple(config.design.chains or sorted(structure.chains))
    missing_chains = set(chains) - structure.chains
    if missing_chains:
        raise StructureValidationError(
            f"Design chain(s) absent from {pdb.name}: {', '.join(sorted(missing_chains))}"
        )
    requested_designable: set[ResidueId] = set()
    for spec in config.design.designable_positions:
        requested_designable.update(parse_position_spec(spec, structure))
    requested_fixed: set[ResidueId] = set()
    for spec in config.design.fixed_positions:
        requested_fixed.update(parse_position_spec(spec, structure))
    if requested_designable & requested_fixed:
        labels = _compact_labels(requested_designable & requested_fixed)
        raise StructureValidationError(f"Residues are both designable and fixed: {labels}")
    wrong_chain = {residue.chain for residue in requested_designable} - set(chains)
    if wrong_chain:
        raise StructureValidationError(
            "Designable positions reference non-design chains: " + ", ".join(sorted(wrong_chain))
        )

    chain_residues = {residue for residue in structure.residues if residue.chain in chains}
    if config.design.designable_positions:
        designable = requested_designable - requested_fixed
    else:
        designable = chain_residues - requested_fixed
    fixed = set(structure.residues) - designable

    fixed_by_chain: dict[str, list[int]] = {}
    for chain in sorted(structure.chains):
        ordered = structure.for_chain(chain)
        fixed_by_chain[chain] = [
            ordinal
            for ordinal, residue in enumerate(ordered, start=1)
            if residue in fixed
        ]
    dictionary = {pdb.stem: fixed_by_chain}

    # ProteinMPNN consumes chain-local 1-based ordinals, not PDB residue numbers.
    # Reconstruct the mutable set from the serialized dictionary and require exact equality.
    assert_mask_equivalence(structure, chains, fixed_by_chain, designable)
    return MaskInspection(
        pdb=pdb,
        structure=structure,
        design_chains=chains,
        designable=frozenset(designable),
        fixed=frozenset(fixed),
        fixed_dictionary=dictionary,
    )


def assert_mask_equivalence(
    structure: StructureIndex,
    design_chains: tuple[str, ...],
    fixed_by_chain: dict[str, list[int]],
    requested_designable: set[ResidueId],
) -> None:
    """Abort unless a ProteinMPNN fixed dictionary preserves the requested mask."""

    reconstructed: set[ResidueId] = set()
    for chain in design_chains:
        fixed_ordinals = set(fixed_by_chain[chain])
        reconstructed.update(
            residue
            for ordinal, residue in enumerate(structure.for_chain(chain), start=1)
            if ordinal not in fixed_ordinals
        )
    if reconstructed != requested_designable:
        raise StructureValidationError(
            "Generated ProteinMPNN fixed-position dictionary inverts or changes the requested mask; aborting"
        )


class ProteinMPNNBackend(ToolBackend):
    name = "proteinmpnn"
    display_name = "ProteinMPNN"
    config_model = ProteinMPNNConfig

    def parse_config(self, raw: dict[str, Any], source: Path) -> ProteinMPNNConfig:
        selected = {
            key: raw[key]
            for key in ("tool", "experiment", "input", "design", "constraints", "resources")
            if key in raw
        }
        config = ProteinMPNNConfig.model_validate(selected)
        if config.input.pdb:
            config.input.pdb = resolve_from_config(config.input.pdb, source)
        if config.input.directory:
            config.input.directory = resolve_from_config(config.input.directory, source)
        return config

    def pdbs(self, config: ProteinMPNNConfig) -> list[Path]:
        if config.input.pdb:
            return [config.input.pdb]
        assert config.input.directory is not None
        if not config.input.directory.is_dir():
            return []
        return sorted(
            path.resolve()
            for path in config.input.directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdb"
        )

    def inspections(self, config: ProteinMPNNConfig) -> list[MaskInspection]:
        return [inspect_mask(config, pdb) for pdb in self.pdbs(config)]

    def validate(self, config: ProteinMPNNConfig) -> ValidationReport:
        report = ValidationReport()
        pdbs = self.pdbs(config)
        if not pdbs:
            source = config.input.pdb or config.input.directory
            report.error(f"No PDB input files found at {source}")
            return report
        stems = [pdb.stem for pdb in pdbs]
        duplicates = {stem for stem in stems if stems.count(stem) > 1}
        if duplicates:
            report.error(f"Duplicate PDB basenames are unsafe: {', '.join(sorted(duplicates))}")
        for pdb in pdbs:
            try:
                inspection = inspect_mask(config, pdb)
                report.details.extend(inspection.render().splitlines())
            except StructureValidationError as exc:
                report.error(str(exc))
            for spec in [
                *config.constraints.omit_by_position,
                *config.constraints.bias_by_position,
            ]:
                try:
                    parse_position_spec(spec, parse_pdb(pdb))
                except StructureValidationError as exc:
                    report.error(str(exc))
        return report

    def build_command(self, config: ProteinMPNNConfig, context: BackendContext) -> CommandPlan:
        executable = context.installation.executable or "protein_mpnn_run.py"
        script = (
            str((context.installation.path / executable).resolve())
            if context.installation.path
            else executable
        )
        inspections = self.inspections(config)
        steps: list[CommandStep] = []
        files: dict[str, str] = {}
        multiple = len(inspections) > 1
        for inspection in inspections:
            slug = re.sub(r"[^A-Za-z0-9_.-]", "_", inspection.pdb.stem)
            fixed_path = context.inputs_dir / f"{slug}.fixed_positions.jsonl"
            files[str(fixed_path)] = json.dumps(inspection.fixed_dictionary) + "\n"
            output_dir = context.output_dir / slug if multiple else context.output_dir
            argv = [
                "python",
                script,
                "--pdb_path",
                str(inspection.pdb),
                "--pdb_path_chains",
                " ".join(inspection.design_chains),
                "--fixed_positions_jsonl",
                str(fixed_path),
                "--out_folder",
                str(output_dir),
                "--num_seq_per_target",
                str(config.design.num_sequences),
                "--sampling_temp",
                " ".join(str(value) for value in config.design.temperatures),
                "--batch_size",
                str(config.design.batch_size),
                "--seed",
                str(config.design.seed),
                "--model_name",
                config.design.model_name,
                "--omit_AAs",
                config.constraints.omit_aas,
            ]
            if config.design.soluble_model:
                argv.append("--use_soluble_model")
            if config.constraints.bias_aas:
                bias_path = context.inputs_dir / f"{slug}.bias_aa.jsonl"
                files[str(bias_path)] = json.dumps(config.constraints.bias_aas) + "\n"
                argv.extend(["--bias_AA_jsonl", str(bias_path)])
            omit_payload = _position_omit_payload(config, inspection)
            if omit_payload:
                omit_path = context.inputs_dir / f"{slug}.omit_aa.jsonl"
                files[str(omit_path)] = json.dumps(omit_payload) + "\n"
                argv.extend(["--omit_AA_jsonl", str(omit_path)])
            bias_payload = _position_bias_payload(config, inspection)
            if bias_payload:
                bias_res_path = context.inputs_dir / f"{slug}.bias_by_res.jsonl"
                files[str(bias_res_path)] = json.dumps(bias_payload) + "\n"
                argv.extend(["--bias_by_res_jsonl", str(bias_res_path)])
            steps.append(
                CommandStep(
                    argv=wrap_environment(argv, context.installation),
                    name=f"design-{slug}",
                    cwd=context.installation.path,
                )
            )
        return CommandPlan(
            steps=steps,
            output_dir=context.output_dir,
            artifacts={
                "files": files,
                "absolute_input_paths": [str(item.pdb) for item in inspections],
            },
        )

    def materialize_artifacts(self, plan: CommandPlan) -> None:
        for raw_path, content in plan.artifacts.get("files", {}).items():
            path = Path(raw_path)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite generated artifact: {path}")
            path.write_text(content, encoding="utf-8")

    def check_environment(self, installation: ToolInstallation) -> EnvironmentCheck:
        return standard_environment_check(
            installation,
            tool=self.name,
            default_executable="protein_mpnn_run.py",
        )


def _chain_ordinals(
    structure: StructureIndex, residues: set[ResidueId] | frozenset[ResidueId]
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {chain: [] for chain in structure.chains}
    for chain in structure.chains:
        result[chain] = [
            ordinal
            for ordinal, residue in enumerate(structure.for_chain(chain), start=1)
            if residue in residues
        ]
    return result


def _position_omit_payload(
    config: ProteinMPNNConfig, inspection: MaskInspection
) -> dict[str, dict[str, list[list[Any]]]] | None:
    if not config.constraints.omit_by_position:
        return None
    by_chain: dict[str, list[list[Any]]] = {chain: [] for chain in inspection.structure.chains}
    for spec, amino_acids in config.constraints.omit_by_position.items():
        selected = parse_position_spec(spec, inspection.structure)
        ordinals = _chain_ordinals(inspection.structure, selected)
        for chain, values in ordinals.items():
            if values:
                by_chain[chain].append([values, amino_acids])
    return {inspection.pdb.stem: by_chain}


def _position_bias_payload(
    config: ProteinMPNNConfig, inspection: MaskInspection
) -> dict[str, dict[str, list[list[float]]]] | None:
    if not config.constraints.bias_by_position:
        return None
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    by_chain: dict[str, list[list[float]]] = {}
    for chain in inspection.structure.chains:
        by_chain[chain] = [
            [0.0 for _ in alphabet] for _ in inspection.structure.for_chain(chain)
        ]
    for spec, biases in config.constraints.bias_by_position.items():
        selected = parse_position_spec(spec, inspection.structure)
        for chain in inspection.structure.chains:
            for ordinal, residue in enumerate(inspection.structure.for_chain(chain), start=1):
                if residue in selected:
                    for amino_acid, value in biases.items():
                        by_chain[chain][ordinal - 1][alphabet.index(amino_acid)] = value
    return {inspection.pdb.stem: by_chain}


def _compact_labels(residues: set[ResidueId] | frozenset[ResidueId]) -> str:
    labels: list[str] = []
    for chain in sorted({item.chain for item in residues}):
        chain_residues = sorted(item for item in residues if item.chain == chain)
        if not chain_residues:
            continue
        start = previous = chain_residues[0]
        for current in chain_residues[1:]:
            contiguous = (
                not previous.insertion_code
                and not current.insertion_code
                and current.number == previous.number + 1
            )
            if contiguous:
                previous = current
                continue
            labels.append(start.label() if start == previous else f"{start.label()}-{previous.label()}")
            start = previous = current
        labels.append(start.label() if start == previous else f"{start.label()}-{previous.label()}")
    return ", ".join(labels)
