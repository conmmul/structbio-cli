"""Adapter for the official CryoZeta inference scripts.

Verified against kiharalab/CryoZeta main on 2026-08-28:

- `inference_demo.sh` takes `-e/--env`, `-g/--gpu`, `-i/--input-json`,
  `-o/--output-dir`, `-m/--mode`, `--checkpoint`, `--interp-checkpoint`,
  `--overwrite`, and validates mode as combined|cryozeta|cryozeta-interpolate.
- `large_inference_demo.sh` takes `-e/--env`, `-g/--gpu`, `-x/--example`,
  `-r/--registration`, `-i/--input-json`, `-o/--output-dir`, `--checkpoint`,
  `--detection-checkpoint`. It has no `--mode` and no `--interp-checkpoint`.
- The input JSON is a list of targets, each with `name`, `modelSeeds`,
  `map_path`, `resolution`, `contour_level`, and `sequences`. A sequence entry
  holds one of `proteinChain`, `dnaSequence`, `rnaSequence`, `ligand`, or
  `ion`; a polymer entry holds `sequence` and `count`, and optionally `msa`,
  `modifications`, and (proteins only) `glycans`.
"""

from __future__ import annotations

import json
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
)
from structbio.validation import (
    StructureValidationError,
    classify_sequence,
    parse_fasta,
    parse_map_header,
)


STANDARD_SCRIPT = "inference_demo.sh"
LARGE_SCRIPT = "large_inference_demo.sh"

# The CryoZeta README directs complexes beyond roughly this size to the large
# pipeline; the standard scripts do not check the size themselves.
LARGE_COMPLEX_RESIDUES = 2800

POLYMER_KEYS = {"protein": "proteinChain", "dna": "dnaSequence", "rna": "rnaSequence"}
SEQUENCE_ENTRY_KEYS = set(POLYMER_KEYS.values()) | {"ligand", "ion"}
REQUIRED_TARGET_KEYS = {
    "name",
    "modelSeeds",
    "map_path",
    "resolution",
    "contour_level",
    "sequences",
}


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class ChainTypes(BaseModel):
    """Which FASTA records are nucleic acids rather than protein."""

    model_config = ConfigDict(extra="forbid")

    protein: list[str] = Field(default_factory=list)
    dna: list[str] = Field(default_factory=list)
    rna: list[str] = Field(default_factory=list)

    def declared(self, name: str) -> str | None:
        for kind in ("protein", "dna", "rna"):
            if name in getattr(self, kind):
                return kind
        return None


class MSASettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    precomputed_msa_dir: Path | None = None
    pairing_db: str | None = None

    def payload(self) -> dict[str, Any]:
        entry: dict[str, Any] = {}
        if self.precomputed_msa_dir:
            entry["precomputed_msa_dir"] = str(self.precomputed_msa_dir)
        if self.pairing_db:
            entry["pairing_db"] = self.pairing_db
        return entry


class CryoInput(BaseModel):
    """Either a native CryoZeta JSON, or a map plus sequences to build one from."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    json_path: Path | None = Field(default=None, alias="json")
    map: Path | None = None
    sequences: Path | None = None
    resolution: float | None = Field(default=None, gt=0, le=30)
    contour_level: float | None = None
    target_name: str | None = None
    chains: ChainTypes = Field(default_factory=ChainTypes)
    msa: MSASettings = Field(default_factory=MSASettings)

    @model_validator(mode="after")
    def one_complete_source(self) -> "CryoInput":
        built = self.map is not None
        if (self.json_path is None) == (not built):
            raise ValueError(
                "Set either input.json (a native CryoZeta JSON) or input.map with "
                "input.sequences, input.resolution, and input.contour_level"
            )
        if built:
            missing = [
                field
                for field in ("sequences", "resolution", "contour_level")
                if getattr(self, field) is None
            ]
            if missing:
                raise ValueError(
                    "Building a CryoZeta target needs input."
                    + ", input.".join(missing)
                )
        return self


class CryoZetaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tool: Literal["cryozeta"]
    experiment: ExperimentSpec
    input: CryoInput
    mode: Literal["combined", "cryozeta", "cryozeta-interpolate"] = "combined"
    large: bool = False
    registration: Literal["auto", "teaser", "svd", "vesper"] = "auto"
    pixi_environment: str | None = None
    gpu_ids: list[int] = Field(default_factory=list)
    checkpoint: Path | None = None
    interpolation_checkpoint: Path | None = None
    detection_checkpoint: Path | None = None
    resources: ResourceConfig = Field(default_factory=lambda: ResourceConfig(gpus=1))

    @model_validator(mode="after")
    def check_selection(self) -> "CryoZetaConfig":
        if any(value < 0 for value in self.gpu_ids):
            raise ValueError("gpu_ids cannot contain negative values")
        if len(set(self.gpu_ids)) != len(self.gpu_ids):
            raise ValueError("gpu_ids cannot contain duplicates")
        if self.large:
            if self.interpolation_checkpoint:
                raise ValueError(
                    "interpolation_checkpoint does not apply to the large-complex "
                    "pipeline; it takes detection_checkpoint instead"
                )
        elif self.detection_checkpoint:
            raise ValueError(
                "detection_checkpoint applies only to the large-complex pipeline; "
                "set large: true to use it"
            )
        return self


def build_targets(config: CryoZetaConfig) -> list[dict[str, Any]]:
    """Turn a map plus a FASTA into CryoZeta's native target list.

    Identical sequences of the same type collapse into one entry with a count,
    which is how CryoZeta expects the copies of a homo-oligomer to be given.
    """

    source = config.input
    assert source.map and source.sequences
    records = parse_fasta(source.sequences)

    ordered: list[tuple[str, str]] = []
    counts: dict[tuple[str, str], int] = {}
    for record in records:
        sequence = "".join(record.chains)
        kind = source.chains.declared(record.name) or classify_sequence(sequence)
        if kind == "ambiguous":
            raise StructureValidationError(
                f"Cannot tell whether {record.name!r} is protein, DNA, or RNA: it uses "
                "only the letters A, C, G and T, which are valid in both. Name it in "
                "input.chains.dna, input.chains.rna, or input.chains.protein"
            )
        key = (kind, sequence)
        if key not in counts:
            ordered.append(key)
        counts[key] = counts.get(key, 0) + 1

    sequences: list[dict[str, Any]] = []
    for kind, sequence in ordered:
        entry: dict[str, Any] = {"sequence": sequence, "count": counts[(kind, sequence)]}
        msa = source.msa.payload()
        if msa and kind in {"protein", "rna"}:
            entry["msa"] = msa
        sequences.append({POLYMER_KEYS[kind]: entry})

    return [
        {
            "name": source.target_name or config.experiment.name,
            "modelSeeds": [],
            "map_path": str(source.map),
            "resolution": source.resolution,
            "contour_level": source.contour_level,
            "sequences": sequences,
        }
    ]


def target_residues(targets: list[dict[str, Any]]) -> int:
    """Total modelled residues across every chain copy of every target."""

    total = 0
    for target in targets:
        for entry in target.get("sequences", []):
            if not isinstance(entry, dict):
                continue
            for key in POLYMER_KEYS.values():
                polymer = entry.get(key)
                if isinstance(polymer, dict) and isinstance(polymer.get("sequence"), str):
                    total += len(polymer["sequence"]) * int(polymer.get("count", 1) or 1)
    return total


class CryoZetaBackend(ToolBackend):
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
                "large",
                "registration",
                "pixi_environment",
                "gpu_ids",
                "checkpoint",
                "interpolation_checkpoint",
                "detection_checkpoint",
                "resources",
            )
            if key in raw
        }
        config = CryoZetaConfig.model_validate(selected)
        for holder, field in (
            (config.input, "json_path"),
            (config.input, "map"),
            (config.input, "sequences"),
            (config.input.msa, "precomputed_msa_dir"),
            (config, "checkpoint"),
            (config, "interpolation_checkpoint"),
            (config, "detection_checkpoint"),
        ):
            value = getattr(holder, field)
            if value is not None:
                setattr(holder, field, resolve_from_config(value, source))
        return config

    def targets(self, config: CryoZetaConfig) -> list[dict[str, Any]]:
        """The target list, whether it was written by hand or built from a map."""

        if config.input.json_path is None:
            return build_targets(config)
        payload = json.loads(config.input.json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise StructureValidationError("CryoZeta input JSON must be a list")
        return payload

    def generates_json(self, config: CryoZetaConfig) -> bool:
        return config.input.json_path is None

    def validate(self, config: CryoZetaConfig) -> ValidationReport:
        report = ValidationReport()
        if self.generates_json(config):
            targets = self._validate_built_input(config, report)
        else:
            targets = self._validate_native_json(config, report)
        if targets is None:
            return report

        residues = target_residues(targets)
        if residues:
            report.details.append(f"Modelled residues: {residues}")
        if residues > LARGE_COMPLEX_RESIDUES and not config.large:
            report.warning(
                f"This complex has about {residues} residues, above the ~"
                f"{LARGE_COMPLEX_RESIDUES} the standard pipeline is meant for. "
                "Add --large to use the large-complex pipeline instead"
            )
        if config.large and residues and residues <= LARGE_COMPLEX_RESIDUES:
            report.warning(
                f"The large-complex pipeline was requested for about {residues} "
                "residues; the standard pipeline is usually faster below "
                f"~{LARGE_COMPLEX_RESIDUES}"
            )

        for label, path in (
            ("checkpoint", config.checkpoint),
            ("interpolation_checkpoint", config.interpolation_checkpoint),
            ("detection_checkpoint", config.detection_checkpoint),
        ):
            if path and not path.is_file():
                report.error(f"CryoZeta {label} does not exist: {path}")
        if config.input.msa.precomputed_msa_dir:
            msa_dir = config.input.msa.precomputed_msa_dir
            if not msa_dir.is_dir():
                report.error(f"precomputed_msa_dir is not a directory: {msa_dir}")

        if len(config.gpu_ids) > 1:
            report.warning(
                "The verified CryoZeta scripts run each inference stage on one GPU"
            )
        report.details.append(f"Targets: {len(targets)}")
        report.details.append(
            f"Pipeline: {LARGE_SCRIPT if config.large else STANDARD_SCRIPT}"
        )
        if not config.large:
            report.details.append(f"Mode: {config.mode}")
        else:
            report.details.append(f"Registration: {config.registration}")
        return report

    def _validate_built_input(
        self, config: CryoZetaConfig, report: ValidationReport
    ) -> list[dict[str, Any]] | None:
        source = config.input
        assert source.map is not None and source.sequences is not None
        try:
            header = parse_map_header(source.map)
            report.details.append(f"Map: {source.map.name}, {header.describe()}")
        except StructureValidationError as exc:
            report.error(str(exc))
            return None
        try:
            targets = build_targets(config)
        except StructureValidationError as exc:
            report.error(str(exc))
            return None

        for entry in targets[0]["sequences"]:
            key, polymer = next(iter(entry.items()))
            copies = f" x{polymer['count']}" if polymer["count"] > 1 else ""
            report.details.append(
                f"  {key}: {len(polymer['sequence'])} residues{copies}"
            )
        if source.contour_level is not None and source.contour_level <= 0:
            report.warning(
                f"contour_level {source.contour_level} is not positive; use the value "
                "recommended for this map by EMDB"
            )
        return targets

    def _validate_native_json(
        self, config: CryoZetaConfig, report: ValidationReport
    ) -> list[dict[str, Any]] | None:
        json_path = config.input.json_path
        assert json_path is not None
        if not json_path.is_file():
            report.error(f"CryoZeta input JSON does not exist: {json_path}")
            return None
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            report.error(f"Invalid CryoZeta input JSON: {exc}")
            return None
        if not isinstance(payload, list) or not payload:
            report.error("CryoZeta input JSON must be a non-empty list")
            return None

        for index, entry in enumerate(payload):
            if not isinstance(entry, dict):
                report.error(f"CryoZeta entry {index} is not an object")
                continue
            missing = REQUIRED_TARGET_KEYS - set(entry)
            if missing:
                report.error(
                    f"CryoZeta entry {index} is missing: {', '.join(sorted(missing))}"
                )
            map_value = entry.get("map_path")
            if isinstance(map_value, str):
                map_path = Path(map_value).expanduser()
                if not map_path.is_absolute():
                    map_path = json_path.parent / map_path
                try:
                    header = parse_map_header(map_path)
                    report.details.append(
                        f"Map for {entry.get('name', index)}: "
                        f"{map_path.name}, {header.describe()}"
                    )
                except StructureValidationError as exc:
                    report.error(str(exc))
            self._validate_sequences(entry, index, report)
        return payload if report.ok else None

    def _validate_sequences(
        self, entry: dict[str, Any], index: int, report: ValidationReport
    ) -> None:
        sequences = entry.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            report.error(f"CryoZeta entry {index} must contain at least one sequence")
            return
        for position, item in enumerate(sequences):
            label = f"CryoZeta entry {index} sequence {position}"
            if not isinstance(item, dict) or len(item) != 1:
                report.error(f"{label} must be an object with exactly one type key")
                continue
            key, value = next(iter(item.items()))
            if key not in SEQUENCE_ENTRY_KEYS:
                report.error(
                    f"{label} has unknown type {key!r}; CryoZeta reads "
                    f"{', '.join(sorted(SEQUENCE_ENTRY_KEYS))}"
                )
                continue
            if not isinstance(value, dict):
                report.error(f"{label} must map {key!r} to an object")
                continue
            if key in POLYMER_KEYS.values():
                if not isinstance(value.get("sequence"), str) or not value.get("sequence"):
                    report.error(f"{label} needs a non-empty 'sequence' string")
                if not isinstance(value.get("count"), int) or value.get("count", 0) < 1:
                    report.error(f"{label} needs a 'count' of at least 1")

    def build_command(self, config: CryoZetaConfig, context: BackendContext) -> CommandPlan:
        default_script = LARGE_SCRIPT if config.large else STANDARD_SCRIPT
        executable = context.installation.executable or default_script
        if config.large and executable == STANDARD_SCRIPT:
            # The configured executable names the standard script; the large
            # pipeline is its sibling in the same checkout.
            executable = LARGE_SCRIPT
        script = (
            str((context.installation.path / executable).resolve())
            if context.installation.path
            else executable
        )

        files: dict[str, str] = {}
        if self.generates_json(config):
            targets_path = context.inputs_dir / "targets.json"
            files[str(targets_path)] = json.dumps(self.targets(config), indent=2) + "\n"
            input_json = targets_path
        else:
            assert config.input.json_path is not None
            input_json = config.input.json_path

        argv = [
            "bash",
            script,
            "--input-json",
            str(input_json),
            "--output-dir",
            str(context.output_dir),
        ]
        if config.large:
            argv.extend(["--registration", config.registration])
        else:
            argv.extend(["--mode", config.mode])
        pixi_environment = config.pixi_environment or context.installation.environment
        if pixi_environment:
            argv.extend(["--env", pixi_environment])
        if config.gpu_ids:
            argv.extend(["--gpu", ",".join(str(value) for value in config.gpu_ids)])
        if config.checkpoint:
            argv.extend(["--checkpoint", str(config.checkpoint)])
        if config.interpolation_checkpoint:
            argv.extend(["--interp-checkpoint", str(config.interpolation_checkpoint)])
        if config.detection_checkpoint:
            argv.extend(["--detection-checkpoint", str(config.detection_checkpoint)])

        # --overwrite is deliberately never generated: structbio only runs into a
        # new or empty output folder, so there is nothing there to re-run over.
        inputs = [path for path in (config.input.map, config.input.sequences) if path]
        if config.input.json_path:
            inputs.append(config.input.json_path)
        for extra in (
            config.checkpoint,
            config.interpolation_checkpoint,
            config.detection_checkpoint,
        ):
            if extra:
                inputs.append(extra)
        return CommandPlan(
            steps=[CommandStep(argv=tuple(argv), cwd=context.installation.path)],
            output_dir=context.output_dir,
            artifacts={
                "files": files,
                "absolute_input_paths": [str(path) for path in inputs],
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
            default_executable=STANDARD_SCRIPT,
            interface=f"interface={STANDARD_SCRIPT} and {LARGE_SCRIPT} (official repository)",
        )
