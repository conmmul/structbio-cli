"""ColabFold backend for the official `colabfold_batch` command."""

from __future__ import annotations

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
    wrap_environment,
)
from structbio.validation import (
    SequenceRecord,
    StructureValidationError,
    parse_fasta,
)


SEQUENCE_SUFFIXES = (".fa", ".fasta", ".a3m", ".csv", ".tsv")
FASTA_SUFFIXES = (".fa", ".fasta")

# ProteinMPNN writes its designed sequences to OUTPUT/seqs/NAME.fa, so a
# ColabFold run can be pointed straight at a ProteinMPNN output folder.
DESIGN_SUBDIRECTORY = "seqs"

REMOTE_MSA_MODES = {"mmseqs2_uniref_env", "mmseqs2_uniref_env_envpair", "mmseqs2_uniref"}


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class ColabFoldInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequences: Path


class ColabFoldMSA(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "mmseqs2_uniref_env",
        "mmseqs2_uniref_env_envpair",
        "mmseqs2_uniref",
        "single_sequence",
    ] = "mmseqs2_uniref_env"
    pair_mode: Literal["unpaired", "paired", "unpaired_paired"] = "unpaired_paired"
    templates: bool = False
    custom_template_path: Path | None = None
    host_url: str | None = None
    only: bool = False


class ColabFoldPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_models: int = Field(default=5, ge=1, le=5)
    num_recycle: int | None = Field(default=None, ge=0, le=100)
    num_seeds: int = Field(default=1, ge=1, le=1_000)
    random_seed: int = Field(default=0, ge=0)
    model_type: Literal[
        "auto",
        "alphafold2",
        "alphafold2_ptm",
        "alphafold2_multimer_v1",
        "alphafold2_multimer_v2",
        "alphafold2_multimer_v3",
        "deepfold_v1",
    ] = "auto"
    stop_at_score: float | None = Field(default=None, ge=0, le=100)
    use_dropout: bool = False


class ColabFoldRelax(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_relax: int = Field(default=0, ge=0, le=5)
    use_gpu: bool = False


class ColabFoldOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: Literal["auto", "plddt", "ptm", "iptm", "multimer"] = "auto"
    save_all: bool = False
    zip_results: bool = False


class ColabFoldConfig(BaseModel):
    """Verified against ColabFold 1.6.2 `colabfold_batch` as of 2026-08-28."""

    model_config = ConfigDict(extra="forbid")

    tool: Literal["colabfold"]
    experiment: ExperimentSpec
    input: ColabFoldInput
    msa: ColabFoldMSA = Field(default_factory=ColabFoldMSA)
    prediction: ColabFoldPrediction = Field(default_factory=ColabFoldPrediction)
    relax: ColabFoldRelax = Field(default_factory=ColabFoldRelax)
    output: ColabFoldOutput = Field(default_factory=ColabFoldOutput)
    resources: ResourceConfig = Field(default_factory=lambda: ResourceConfig(gpus=1))

    @model_validator(mode="after")
    def check_combinations(self) -> "ColabFoldConfig":
        if self.relax.num_relax > self.prediction.num_models:
            raise ValueError(
                "relax.num_relax cannot exceed prediction.num_models: ColabFold relaxes "
                "the top ranked models only"
            )
        if self.relax.use_gpu and not self.relax.num_relax:
            raise ValueError("relax.use_gpu has no effect unless relax.num_relax is at least 1")
        if self.msa.custom_template_path and not self.msa.templates:
            raise ValueError("msa.custom_template_path requires msa.templates: true")
        return self


def _fastas_in(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        item
        for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in FASTA_SUFFIXES
    )


def design_folders(path: Path) -> list[Path]:
    """Find the per-structure `seqs` folders of a ProteinMPNN directory batch.

    ProteinMPNN run over a folder of backbones writes OUTPUT/STRUCTURE/seqs/*.fa,
    one subfolder per input structure, so the sequences are one level deeper than
    a single-structure run.
    """

    if not path.is_dir():
        return []
    return [
        child / DESIGN_SUBDIRECTORY
        for child in sorted(path.iterdir())
        if child.is_dir() and _fastas_in(child / DESIGN_SUBDIRECTORY)
    ]


def resolve_sequence_source(path: Path) -> Path:
    """Accept a ProteinMPNN output folder by using the `seqs` folder inside it."""

    if not path.is_dir():
        return path
    has_own = any(
        item.suffix.lower() in SEQUENCE_SUFFIXES for item in path.iterdir() if item.is_file()
    )
    if has_own:
        return path
    if _fastas_in(path / DESIGN_SUBDIRECTORY):
        return path / DESIGN_SUBDIRECTORY
    return path


class ColabFoldBackend(ToolBackend):
    """Adapter for ColabFold 1.6.2; `colabfold_batch INPUT RESULTS`."""

    name = "colabfold"
    display_name = "ColabFold"
    config_model = ColabFoldConfig

    def parse_config(self, raw: dict[str, Any], source: Path) -> ColabFoldConfig:
        selected = {
            key: raw[key]
            for key in (
                "tool",
                "experiment",
                "input",
                "msa",
                "prediction",
                "relax",
                "output",
                "resources",
            )
            if key in raw
        }
        config = ColabFoldConfig.model_validate(selected)
        config.input.sequences = resolve_from_config(config.input.sequences, source)
        if config.msa.custom_template_path:
            config.msa.custom_template_path = resolve_from_config(
                config.msa.custom_template_path, source
            )
        return config

    def sequence_source(self, config: ColabFoldConfig) -> Path:
        return resolve_sequence_source(config.input.sequences)

    def batch_folders(self, config: ColabFoldConfig) -> list[Path]:
        """The `seqs` folders of a ProteinMPNN directory batch, if that is the input."""

        if self.sequence_source(config) != config.input.sequences:
            return []
        return design_folders(config.input.sequences)

    def fasta_files(self, config: ColabFoldConfig) -> list[Path]:
        batch = self.batch_folders(config)
        if batch:
            return [fasta for folder in batch for fasta in _fastas_in(folder)]
        source = self.sequence_source(config)
        if source.is_dir():
            return _fastas_in(source)
        return [source] if source.suffix.lower() in FASTA_SUFFIXES else []

    def merged_fasta_text(self, config: ColabFoldConfig) -> str:
        """Concatenate a batch's FASTA files verbatim; sequences are never rewritten."""

        blocks: list[str] = []
        for fasta in self.fasta_files(config):
            text = fasta.read_text(encoding="utf-8")
            blocks.append(text if text.endswith("\n") else text + "\n")
        return "".join(blocks)

    def validate(self, config: ColabFoldConfig) -> ValidationReport:
        report = ValidationReport()
        given = config.input.sequences
        if not given.exists():
            report.error(f"Sequence input does not exist: {given}")
            return report

        source = self.sequence_source(config)
        batch = self.batch_folders(config)
        if batch:
            report.details.append(
                f"Using designed sequences from {len(batch)} ProteinMPNN result folders "
                f"below {given}"
            )
        elif source != given:
            report.details.append(f"Using designed sequences from {source}")
        if batch:
            report.details.append(f"Input files: {len(self.fasta_files(config))}")
        elif source.is_dir():
            accepted = sorted(
                item
                for item in source.iterdir()
                if item.is_file() and item.suffix.lower() in SEQUENCE_SUFFIXES
            )
            if not accepted:
                report.error(f"No {', '.join(SEQUENCE_SUFFIXES)} files found in {source}")
                return report
            report.details.append(f"Input files: {len(accepted)}")
        elif source.suffix.lower() not in SEQUENCE_SUFFIXES:
            report.error(
                f"Unsupported sequence input {source.name}; ColabFold reads "
                f"{', '.join(SEQUENCE_SUFFIXES)}"
            )
            return report

        fasta_files = self.fasta_files(config)
        records: list[SequenceRecord] = []
        for fasta in fasta_files:
            try:
                records.extend(parse_fasta(fasta))
            except StructureValidationError as exc:
                report.error(str(exc))
        if not fasta_files:
            report.warning(
                f"{source.name} is not FASTA, so its sequences were not checked here; "
                "ColabFold validates its own CSV and a3m input"
            )

        if records:
            names = [record.name for record in records]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                report.warning(
                    "Duplicate sequence names share one ColabFold job name and overwrite "
                    f"each other: {', '.join(duplicates[:5])}"
                )
            complexes = [record for record in records if len(record.chains) > 1]
            longest = max(record.length for record in records)
            report.details.append(f"Sequences: {len(records)}")
            if complexes:
                report.details.append(f"Complexes: {len(complexes)}")
            report.details.append(f"Longest sequence: {longest} residues")
            if longest > 1500:
                report.warning(
                    f"The longest sequence is {longest} residues; expect heavy GPU memory use"
                )
            model_type = config.prediction.model_type
            if complexes and model_type != "auto" and "multimer" not in model_type:
                report.warning(
                    f"Input contains complexes but prediction.model_type is {model_type}; "
                    "'auto' selects a multimer model instead"
                )

        if config.msa.mode in REMOTE_MSA_MODES and not config.msa.host_url:
            report.warning(
                "MSAs will be built by the public ColabFold MMseqs2 server, so every input "
                "sequence leaves this machine. Use msa.mode: single_sequence, or set "
                "msa.host_url to a server you run, to keep sequences local"
            )
        if config.msa.custom_template_path and not config.msa.custom_template_path.is_dir():
            report.error(
                f"msa.custom_template_path is not a directory: {config.msa.custom_template_path}"
            )

        report.details.append(f"Models per sequence: {config.prediction.num_models}")
        report.details.append(f"MSA mode: {config.msa.mode}")
        if config.relax.num_relax:
            report.details.append(f"Amber relax: top {config.relax.num_relax}")
        return report

    def build_command(self, config: ColabFoldConfig, context: BackendContext) -> CommandPlan:
        executable = context.installation.executable or "colabfold_batch"
        program = (
            str((context.installation.path / executable).resolve())
            if context.installation.path
            else executable
        )
        files: dict[str, str] = {}
        if self.batch_folders(config):
            # colabfold_batch takes one input path, and loading its models once for
            # the whole batch is far faster than once per design folder.
            merged = context.inputs_dir / "sequences.fa"
            files[str(merged)] = self.merged_fasta_text(config)
            source = merged
        else:
            source = self.sequence_source(config)
        argv = [program, str(source), str(context.output_dir)]

        argv.extend(["--num-models", str(config.prediction.num_models)])
        if config.prediction.num_recycle is not None:
            argv.extend(["--num-recycle", str(config.prediction.num_recycle)])
        if config.prediction.num_seeds != 1:
            argv.extend(["--num-seeds", str(config.prediction.num_seeds)])
        if config.prediction.random_seed:
            argv.extend(["--random-seed", str(config.prediction.random_seed)])
        if config.prediction.model_type != "auto":
            argv.extend(["--model-type", config.prediction.model_type])
        if config.prediction.stop_at_score is not None:
            argv.extend(["--stop-at-score", str(config.prediction.stop_at_score)])
        if config.prediction.use_dropout:
            argv.append("--use-dropout")

        argv.extend(["--msa-mode", config.msa.mode])
        if config.msa.pair_mode != "unpaired_paired":
            argv.extend(["--pair-mode", config.msa.pair_mode])
        if config.msa.templates:
            argv.append("--templates")
        if config.msa.custom_template_path:
            argv.extend(["--custom-template-path", str(config.msa.custom_template_path)])
        if config.msa.host_url:
            argv.extend(["--host-url", config.msa.host_url])
        if config.msa.only:
            argv.append("--msa-only")

        if config.relax.num_relax:
            argv.extend(["--num-relax", str(config.relax.num_relax)])
            if config.relax.use_gpu:
                argv.append("--use-gpu-relax")

        if config.output.rank != "auto":
            argv.extend(["--rank", config.output.rank])
        if config.output.save_all:
            argv.append("--save-all")
        if config.output.zip_results:
            argv.append("--zip")

        # --overwrite-existing-results is deliberately never generated: structbio
        # only ever runs into a new or empty output folder.
        return CommandPlan(
            steps=[
                CommandStep(
                    argv=wrap_environment(argv, context.installation),
                    cwd=context.installation.path,
                )
            ],
            output_dir=context.output_dir,
            artifacts={
                "files": files,
                "absolute_input_paths": [
                    str(path) for path in (self.fasta_files(config) or [source])
                ],
            },
        )

    def materialize_artifacts(self, plan: CommandPlan) -> None:
        for raw_path, content in plan.artifacts.get("files", {}).items():
            path = Path(raw_path)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite generated artifact: {path}")
            path.write_text(content, encoding="utf-8")

    def check_environment(self, installation: ToolInstallation) -> EnvironmentCheck:
        executable = executable_path(installation)
        manager_ok = installation.manager != "conda" or shutil.which("conda") is not None
        details = ["interface=colabfold_batch (ColabFold 1.6.2)"]
        if installation.environment:
            details.append(f"environment={installation.environment}")
        if not manager_ok:
            details.append("conda executable not found")
        return EnvironmentCheck(
            configured=installation.path is not None or executable is not None,
            found=bool(executable and manager_ok),
            executable=str(executable) if executable else None,
            details=tuple(details),
        )
