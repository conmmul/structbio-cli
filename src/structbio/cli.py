"""Typer command-line interface for structbio."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from structbio import __version__, quick, wrappers
from structbio.config import (
    USER_CONFIG_TEMPLATE,
    LoadedConfig,
    lab_config_path,
    load_command_line_config,
    load_config,
    load_settings,
    parse_cli_overrides,
    user_config_path,
)
from structbio.environment import detect_gpu, detect_unwrapped_tools, select_idle_gpu
from structbio.execution import execute_plan, plan_input_paths
from structbio.experiment import (
    ExperimentManager,
    ExperimentPaths,
    direct_paths,
    prepare_output_dir,
    read_metadata,
    safe_name,
    update_metadata,
    write_records,
)
from structbio.slurm import generate_slurm_script
from structbio.tools import get_backend, get_backends
from structbio.tools.base import BackendContext, CommandPlan, ToolBackend, ValidationReport
from structbio.tools.proteinmpnn import ProteinMPNNBackend, ProteinMPNNConfig


app = typer.Typer(
    name="structbio",
    help=(
        "Short commands for structural-biology software on a workstation.\n\n"
        "Quick form:  structbio TOOL RUNTYPE ... OUTPUT_FOLDER\n"
        "YAML form:   structbio TOOL run CONFIG.yaml"
    ),
    no_args_is_help=True,
)
rfdiffusion_app = typer.Typer(
    help=(
        "RFdiffusion backbone design.\n\n"
        "  rfdiffusion monomer LENGTH OUTPUT\n"
        "  rfdiffusion symmetry GROUP TOTAL_LENGTH OUTPUT\n"
        "  rfdiffusion binder TARGET.pdb LENGTH OUTPUT\n"
        "  rfdiffusion partial INPUT.pdb STEPS OUTPUT"
    ),
    no_args_is_help=True,
)
proteinmpnn_app = typer.Typer(
    help=(
        "ProteinMPNN sequence design.\n\n"
        "  proteinmpnn design INPUT.pdb NUM_SEQUENCES OUTPUT"
    ),
    no_args_is_help=True,
)
colabfold_app = typer.Typer(
    help=(
        "ColabFold structure prediction.\n\n"
        "  colabfold predict SEQUENCES OUTPUT"
    ),
    no_args_is_help=True,
)
cryozeta_app = typer.Typer(
    help=(
        "CryoZeta cryo-EM structure modelling.\n\n"
        "  cryozeta predict MAP.mrc CHAINS.fasta OUTPUT --resolution R --contour C\n"
        "  cryozeta predict-json TARGETS.json OUTPUT"
    ),
    no_args_is_help=True,
)
app.add_typer(rfdiffusion_app, name="rfdiffusion")
app.add_typer(proteinmpnn_app, name="proteinmpnn")
app.add_typer(colabfold_app, name="colabfold")
app.add_typer(cryozeta_app, name="cryozeta")


OUTPUT_HELP = "Output folder; its name is also used as the output file prefix"
GPU_HELP = "GPU to use: an id such as 0, several as 0,1, or 'auto' for the idlest"
QUIET_HELP = "Do not echo the tool's output; it is still written to the log"
SET_HELP = "Override any configuration value: dotted.key=YAML_VALUE"


def _abort(message: str) -> None:
    typer.echo(f"Error: {message}", err=True)
    raise typer.Exit(2)


def _load_backend_config(
    config_path: Path,
    expected_tool: str | None = None,
    set_values: list[str] | None = None,
) -> tuple[LoadedConfig, ToolBackend, Any]:
    try:
        loaded = load_config(config_path, overrides=parse_cli_overrides(set_values or []))
        return _backend_for(loaded, expected_tool)
    except (ValueError, ValidationError) as exc:
        _abort(str(exc))
    raise AssertionError("unreachable")


def _backend_for(
    loaded: LoadedConfig, expected_tool: str | None
) -> tuple[LoadedConfig, ToolBackend, Any]:
    tool = loaded.data.get("tool")
    if not isinstance(tool, str):
        _abort("Configuration must contain a string 'tool' field")
    if expected_tool and tool != expected_tool:
        _abort(f"Expected tool: {expected_tool}; configuration selects: {tool}")
    backend = get_backend(tool)
    config = backend.parse_config(loaded.data, loaded.source)
    return loaded, backend, config


def _show_report(report: ValidationReport) -> None:
    for detail in report.details:
        typer.echo(detail)
    for warning in report.warnings:
        typer.echo(f"WARNING: {warning}")
    for error in report.errors:
        typer.echo(f"ERROR: {error}")


def _validated(
    config_path: Path,
    expected_tool: str | None = None,
    set_values: list[str] | None = None,
) -> tuple[LoadedConfig, ToolBackend, Any, ValidationReport]:
    loaded, backend, config = _load_backend_config(config_path, expected_tool, set_values)
    report = backend.validate(config)
    if not report.ok:
        _show_report(report)
        raise typer.Exit(2)
    return loaded, backend, config, report


def _manager(loaded: LoadedConfig) -> ExperimentManager:
    root = loaded.settings.experiments_root
    if not root.is_absolute():
        root = Path.cwd() / root
    return ExperimentManager(root)


def _installation(loaded: LoadedConfig, backend: ToolBackend) -> Any:
    installation = loaded.settings.tools.get(backend.name)
    if installation is None:
        _abort(
            f"No installation is configured for {backend.display_name}. "
            f"Run 'structbio setup' and edit {user_config_path()}"
        )
    return installation


def _context(
    loaded: LoadedConfig, backend: ToolBackend, paths: ExperimentPaths
) -> BackendContext:
    return BackendContext(
        source=loaded.source,
        installation=_installation(loaded, backend),
        experiment_dir=paths.root,
        output_dir=paths.outputs,
        inputs_dir=paths.inputs,
    )


def _require_installed(loaded: LoadedConfig, backend: ToolBackend) -> Any:
    installation = _installation(loaded, backend)
    environment = backend.check_environment(installation)
    if not environment.found:
        details = "; ".join(environment.details)
        _abort(
            f"{backend.display_name} is not available on this machine. {details} "
            f"Check paths with 'structbio doctor'."
        )
    return installation


def _resolve_gpu(gpu: str | None) -> str | None:
    """Turn --gpu into a concrete id list, resolving 'auto' against this machine."""

    if not gpu:
        return None
    if gpu.strip().lower() == "auto":
        index = select_idle_gpu()
        if index is None:
            _abort("--gpu auto needs nvidia-smi to see the GPUs; name an id such as --gpu 0")
        typer.echo(f"Using GPU {index}, which has the most free memory.")
        return str(index)
    try:
        return ",".join(str(item) for item in quick.parse_gpu_ids(gpu))
    except ValueError as exc:
        _abort(str(exc))
    raise AssertionError("unreachable")


def _cryozeta_gpu(gpu: str | None) -> str | None:
    """CryoZeta takes GPU ids on its own command line rather than through the environment."""

    return _resolve_gpu(gpu)


def _with_gpu(plan: CommandPlan, gpu: str | None) -> CommandPlan:
    value = _resolve_gpu(gpu)
    if value is None:
        return plan
    plan.steps = [
        dataclasses.replace(step, env={**step.env, "CUDA_VISIBLE_DEVICES": value})
        for step in plan.steps
    ]
    return plan


def _preview_plan(
    loaded: LoadedConfig, backend: ToolBackend, config: Any
) -> tuple[Path, CommandPlan]:
    experiment_dir = _manager(loaded).candidate(config.experiment.name)
    paths = ExperimentManager.paths(experiment_dir)
    return experiment_dir, backend.build_command(config, _context(loaded, backend, paths))


# ---------------------------------------------------------------------------
# Quick commands: positional arguments, plain output folders
# ---------------------------------------------------------------------------


def _quick_run(
    tool: str,
    fragment_builder: Any,
    output: Path,
    *,
    gpu: str | None,
    dry_run: bool,
    set_values: list[str],
    quiet: bool = False,
    **arguments: Any,
) -> None:
    """Validate and run a positional command against a plain output folder."""

    output_dir = output.expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    try:
        name = safe_name(output_dir.name)
    except ValueError:
        _abort(f"Output folder name must contain letters or numbers: {output_dir.name!r}")
    try:
        fragment = fragment_builder(name=name, **arguments)
        loaded = load_command_line_config(
            fragment, overrides=parse_cli_overrides(set_values)
        )
        loaded, backend, config = _backend_for(loaded, tool)
        report = backend.validate(config)
    except (ValueError, ValidationError) as exc:
        _abort(str(exc))
    if not report.ok:
        _show_report(report)
        raise typer.Exit(2)

    if dry_run:
        plan = _with_gpu(
            backend.build_command(config, _context(loaded, backend, direct_paths(output_dir))),
            gpu,
        )
        _show_report(report)
        typer.echo(f"\nCommand:\n{plan.render()}")
        typer.echo(f"\nOutput folder: {output_dir}")
        typer.echo("Dry run: nothing was created or executed.")
        return

    installation = _require_installed(loaded, backend)
    try:
        paths = prepare_output_dir(output_dir)
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    plan = _with_gpu(backend.build_command(config, _context(loaded, backend, paths)), gpu)
    backend.materialize_artifacts(plan)
    write_records(
        paths,
        config=loaded.data,
        command=plan.render(),
        tool_name=backend.name,
        tool_path=installation.path,
        input_paths=plan_input_paths(plan),
        status="prepared",
    )
    typer.echo(f"{backend.display_name}: {plan.render()}")
    typer.echo(f"Output folder: {paths.outputs}")
    return_code = execute_plan(plan, paths, stream=not quiet)
    if return_code:
        typer.echo(f"Failed with exit code {return_code}; see {paths.stderr}", err=True)
        raise typer.Exit(return_code)
    typer.echo(f"Done. Results are in {paths.outputs}")


@rfdiffusion_app.command("monomer")
def rfdiffusion_monomer(
    length: int = typer.Argument(..., min=1, help="Residues per design"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    num: int = typer.Option(1, "-n", "--num", min=1, help="Number of designs"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Design new monomer backbones from nothing but a length.

    Example: rfdiffusion monomer 150 my_monomers -n 10
    """

    _quick_run(
        "rfdiffusion",
        quick.rfdiffusion_monomer,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        length=length,
        num_designs=num,
    )


@rfdiffusion_app.command("symmetry")
def rfdiffusion_symmetry(
    group: str = typer.Argument(..., help="Symmetry group: c2..cN, d2..dN, or tetrahedral"),
    length: int = typer.Argument(..., min=1, help="Total residues across every subunit"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    num: int = typer.Option(1, "-n", "--num", min=1, help="Number of designs"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Design symmetric oligomers.

    The length is the total across all subunits and must divide by the subunit
    count. Example: rfdiffusion symmetry c4 400 my_tetramers
    """

    _quick_run(
        "rfdiffusion",
        quick.rfdiffusion_symmetry,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        symmetry=group,
        length=length,
        num_designs=num,
    )


@rfdiffusion_app.command("binder")
def rfdiffusion_binder(
    target: Path = typer.Argument(..., exists=True, dir_okay=False, help="Target structure"),
    length: int = typer.Argument(..., min=1, help="Residues in the designed binder"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    chain: str | None = typer.Option(
        None, "--chain", help="Target chain; required when the file has several"
    ),
    hotspots: str | None = typer.Option(
        None, "--hotspots", help="Target residues to engage, such as B30,B33,B34"
    ),
    num: int = typer.Option(1, "-n", "--num", min=1, help="Number of designs"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Design binders against a target structure.

    The target contig is read from the residue numbering in the file itself.
    Example: rfdiffusion binder target.pdb 100 my_binders --hotspots B30,B33
    """

    _quick_run(
        "rfdiffusion",
        quick.rfdiffusion_binder,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        target=target,
        length=length,
        num_designs=num,
        chain=chain,
        hotspots=hotspots,
    )


@rfdiffusion_app.command("partial")
def rfdiffusion_partial(
    pdb: Path = typer.Argument(..., exists=True, dir_okay=False, help="Structure to diversify"),
    steps: int = typer.Argument(..., min=1, help="Partial diffusion timesteps"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    num: int = typer.Option(1, "-n", "--num", min=1, help="Number of designs"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Diversify an existing structure by partial diffusion.

    Every chain is kept at its own length. Fewer steps stay closer to the input.
    Example: rfdiffusion partial start.pdb 10 my_variants -n 20
    """

    _quick_run(
        "rfdiffusion",
        quick.rfdiffusion_partial,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        pdb=pdb,
        steps=steps,
        num_designs=num,
    )


@proteinmpnn_app.command("design")
def proteinmpnn_design(
    input_path: Path = typer.Argument(..., exists=True, help="PDB file, or a folder of PDBs"),
    num_sequences: int = typer.Argument(..., min=1, help="Sequences per structure"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    chains: str | None = typer.Option(None, "--chains", help="Chains to design, such as A,B"),
    designable: str | None = typer.Option(
        None, "--designable", help="Only these positions may change, such as A:697-749"
    ),
    fixed: str | None = typer.Option(
        None, "--fixed", help="Positions that must not change, such as A:700,A:705"
    ),
    temperature: str | None = typer.Option(
        None, "--temp", help="Sampling temperature(s), default 0.1"
    ),
    seed: int = typer.Option(0, "--seed", min=0, help="Random seed"),
    soluble: bool = typer.Option(False, "--soluble", help="Use the soluble-protein model"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Design sequences for a structure, or for every PDB in a folder.

    Without --designable every residue of the selected chains may change.
    Example: proteinmpnn design 7kdp.pdb 8 my_sequences --designable A:697-749
    """

    _quick_run(
        "proteinmpnn",
        quick.proteinmpnn_design,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        input_path=input_path,
        num_sequences=num_sequences,
        chains=chains,
        designable=designable,
        fixed=fixed,
        temperature=temperature,
        seed=seed,
        soluble=soluble,
    )


@colabfold_app.command("predict")
def colabfold_predict(
    sequences: Path = typer.Argument(
        ..., exists=True, help="FASTA file, a folder of them, or a ProteinMPNN output folder"
    ),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    num_models: int = typer.Option(
        5, "-n", "--num-models", min=1, max=5, help="Models per sequence"
    ),
    msa_mode: str = typer.Option(
        "mmseqs2_uniref_env",
        "--msa-mode",
        help="mmseqs2_uniref_env, mmseqs2_uniref_env_envpair, mmseqs2_uniref, or single_sequence",
    ),
    templates: bool = typer.Option(False, "--templates", help="Use PDB templates"),
    relax: int = typer.Option(
        0, "--relax", min=0, max=5, help="Amber-relax this many top-ranked models"
    ),
    recycle: int | None = typer.Option(None, "--recycle", min=0, help="Recycle iterations"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Predict structures for designed or natural sequences.

    Point it at a ProteinMPNN output folder and it finds the designed sequences
    itself. By default MSAs come from the public ColabFold server, which means
    the sequences leave this machine; --msa-mode single_sequence keeps them local.

    Example: colabfold predict my_sequences my_folds -n 5 --relax 1
    """

    _quick_run(
        "colabfold",
        quick.colabfold_predict,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        sequences=sequences,
        num_models=num_models,
        msa_mode=msa_mode,
        templates=templates,
        relax=relax,
        num_recycle=recycle,
    )


@cryozeta_app.command("predict")
def cryozeta_predict(
    density_map: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Cryo-EM density map (.map, .mrc, .map.gz)"
    ),
    sequences: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="FASTA of every chain in the complex"
    ),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    resolution: float = typer.Option(
        ..., "--resolution", min=0.1, max=30.0, help="Map resolution in angstroms"
    ),
    contour: float = typer.Option(
        ..., "--contour", help="Recommended contour level for the map"
    ),
    mode: str = typer.Option(
        "combined", "--mode", help="combined, cryozeta, or cryozeta-interpolate"
    ),
    large: bool = typer.Option(
        False, "--large", help="Use the large-complex pipeline (above ~2800 residues)"
    ),
    registration: str = typer.Option(
        "auto", "--registration", help="Large-complex registration: auto, teaser, svd, vesper"
    ),
    dna: str | None = typer.Option(None, "--dna", help="FASTA record names that are DNA"),
    rna: str | None = typer.Option(None, "--rna", help="FASTA record names that are RNA"),
    protein: str | None = typer.Option(
        None, "--protein", help="FASTA record names that are protein, when ambiguous"
    ),
    msa_dir: Path | None = typer.Option(
        None, "--msa-dir", help="Directory of precomputed .a3m files"
    ),
    pairing_db: str | None = typer.Option(
        None, "--pairing-db", help="MSA pairing database, such as uniref100"
    ),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Model a structure into a cryo-EM map from the map and its sequences.

    The native CryoZeta target JSON is written for you into
    OUTPUT/.structbio/inputs/targets.json. Identical sequences are counted as
    copies of one chain. Ligands, ions, and modifications need a hand-written
    JSON and `cryozeta predict-json`.

    Example: cryozeta predict emd_44046.map.gz chains.fasta my_model
             --resolution 2.99 --contour 0.3
    """

    _quick_run(
        "cryozeta",
        quick.cryozeta_predict,
        output,
        gpu=None,  # CryoZeta takes GPU ids on its own command line
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        density_map=density_map,
        sequences=sequences,
        resolution=resolution,
        contour=contour,
        mode=mode,
        large=large,
        registration=registration,
        dna=dna,
        rna=rna,
        protein=protein,
        msa_dir=msa_dir,
        pairing_db=pairing_db,
        gpu_ids=_cryozeta_gpu(gpu),
    )


@cryozeta_app.command("predict-json")
def cryozeta_predict_json(
    targets: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Native CryoZeta target JSON"
    ),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    mode: str = typer.Option(
        "combined", "--mode", help="combined, cryozeta, or cryozeta-interpolate"
    ),
    large: bool = typer.Option(
        False, "--large", help="Use the large-complex pipeline (above ~2800 residues)"
    ),
    registration: str = typer.Option(
        "auto", "--registration", help="Large-complex registration: auto, teaser, svd, vesper"
    ),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Run a CryoZeta target list you wrote yourself.

    Use this for anything the short form cannot describe: several targets in one
    run, ligands and ions, chain modifications, or glycans.

    Example: cryozeta predict-json targets.json my_models --gpu 0
    """

    _quick_run(
        "cryozeta",
        quick.cryozeta_predict_json,
        output,
        gpu=None,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        input_json=targets,
        mode=mode,
        large=large,
        registration=registration,
        gpu_ids=_cryozeta_gpu(gpu),
    )


# ---------------------------------------------------------------------------
# YAML commands: full control over every documented option
# ---------------------------------------------------------------------------


def _validate_command(
    config_path: Path, tool: str | None = None, set_values: list[str] | None = None
) -> None:
    _, backend, _, report = _validated(config_path, tool, set_values)
    _show_report(report)
    typer.echo(f"\n{backend.display_name} configuration is valid.")


def _command_command(config_path: Path, tool: str, set_values: list[str] | None = None) -> None:
    loaded, backend, config, _ = _validated(config_path, tool, set_values)
    experiment_dir, plan = _preview_plan(loaded, backend, config)
    typer.echo(plan.render())
    typer.echo(f"\nOutput: {experiment_dir / 'outputs'}")


def _run_command(
    config_path: Path,
    tool: str,
    dry_run: bool,
    set_values: list[str] | None = None,
    gpu: str | None = None,
    quiet: bool = False,
) -> None:
    loaded, backend, config, report = _validated(config_path, tool, set_values)
    if dry_run:
        experiment_dir, plan = _preview_plan(loaded, backend, config)
        _show_report(report)
        typer.echo(f"\nCommand:\n{_with_gpu(plan, gpu).render()}")
        typer.echo(f"\nExperiment: {experiment_dir}")
        typer.echo(f"Output: {plan.output_dir}")
        typer.echo("Dry run: nothing was created or executed.")
        return

    installation = _require_installed(loaded, backend)
    paths = _manager(loaded).create(config.experiment.name)
    plan = _with_gpu(backend.build_command(config, _context(loaded, backend, paths)), gpu)
    backend.materialize_artifacts(plan)
    write_records(
        paths,
        config=loaded.data,
        command=plan.render(),
        tool_name=backend.name,
        tool_path=installation.path,
        input_paths=plan_input_paths(plan),
        status="prepared",
    )
    typer.echo(f"Experiment: {paths.root}")
    return_code = execute_plan(plan, paths, stream=not quiet)
    if return_code:
        typer.echo(f"Execution failed with exit code {return_code}; see {paths.stderr}", err=True)
        raise typer.Exit(return_code)
    typer.echo(f"Completed. Outputs: {paths.outputs}")


def _submit_command(
    config_path: Path,
    tool: str,
    dry_run: bool,
    execute: bool,
    set_values: list[str] | None = None,
) -> None:
    loaded, backend, config, report = _validated(config_path, tool, set_values)
    manager = _manager(loaded)
    if dry_run:
        experiment_dir, plan = _preview_plan(loaded, backend, config)
        paths = manager.paths(experiment_dir)
    else:
        paths = manager.create(config.experiment.name)
        plan = backend.build_command(config, _context(loaded, backend, paths))

    resources = config.resources
    profile = None
    if resources.cluster:
        profile = loaded.settings.cluster_profiles.get(resources.cluster)
        if profile is None:
            _abort(f"Unknown cluster profile: {resources.cluster}")
    script = generate_slurm_script(
        plan,
        experiment_dir=paths.root,
        job_name=config.experiment.name,
        resources=resources,
        profile=profile,
    )
    if dry_run:
        _show_report(report)
        typer.echo(script)
        typer.echo(f"Experiment: {paths.root}")
        typer.echo("Dry run: nothing was created, executed, or submitted.")
        return

    backend.materialize_artifacts(plan)
    paths.slurm_script.write_text(script, encoding="utf-8")
    installation = _installation(loaded, backend)
    write_records(
        paths,
        config=loaded.data,
        command=plan.render(),
        tool_name=backend.name,
        tool_path=installation.path,
        input_paths=plan_input_paths(plan),
        status="script-generated",
    )
    typer.echo(f"SLURM script: {paths.slurm_script}")
    if not execute:
        typer.echo("Not submitted. Re-run with --execute to invoke sbatch explicitly.")
        return
    if shutil.which("sbatch") is None:
        _abort("sbatch is not available")
    result = subprocess.run(
        ["sbatch", str(paths.slurm_script)], capture_output=True, text=True, check=False
    )
    if result.returncode:
        update_metadata(paths, status="submission-failed", submission_error=result.stderr.strip())
        _abort(result.stderr.strip() or "sbatch failed")
    output = result.stdout.strip()
    job_id = output.rsplit(maxsplit=1)[-1] if output else None
    update_metadata(paths, status="submitted", slurm_job_id=job_id, sbatch_output=output)
    typer.echo(output)


def _register_yaml_commands(subapp: typer.Typer, tool: str) -> None:
    @subapp.command("validate")
    def validate_tool(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
    ) -> None:
        """Check a YAML configuration and every structure selection in it."""

        _validate_command(config, tool, set_value)

    @subapp.command("command")
    def command_tool(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
    ) -> None:
        """Print the exact upstream command a YAML configuration produces."""

        _command_command(config, tool, set_value)

    @subapp.command("run")
    def run_tool(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate and plan only"),
        gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
        quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
        set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
    ) -> None:
        """Run a YAML configuration into a dated experiment folder."""

        _run_command(config, tool, dry_run, set_value, gpu, quiet)

    @subapp.command("submit")
    def submit_tool(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print script only"),
        execute: bool = typer.Option(False, "--execute", help="Explicitly invoke sbatch"),
        set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
    ) -> None:
        """Optional: write a SLURM script for a shared cluster; not used on a workstation."""

        _submit_command(config, tool, dry_run, execute, set_value)


# ---------------------------------------------------------------------------
# Setup and diagnostics
# ---------------------------------------------------------------------------


@app.command("setup")
def setup(
    bin_dir: Path = typer.Option(
        wrappers.DEFAULT_WRAPPER_DIR, "--bin-dir", help="Where tool commands are written"
    ),
    wrappers_only: bool = typer.Option(
        False, "--wrappers-only", help="Do not touch the configuration file"
    ),
) -> None:
    """Create the workstation configuration file and the per-tool commands."""

    config_path = user_config_path()
    if not wrappers_only:
        if config_path.exists():
            typer.echo(f"Configuration already exists: {config_path}")
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(USER_CONFIG_TEMPLATE, encoding="utf-8")
            typer.echo(f"Wrote {config_path}")
        typer.echo("Edit it so each path points at the software installed on this machine.\n")

    for path, state in wrappers.install_wrappers(bin_dir):
        typer.echo(f"{state:<32} {path}")
    typer.echo("\nThen check the installation with: structbio doctor")
    _warn_about_path(bin_dir)


def _warn_about_path(bin_dir: Path) -> None:
    """Say so, unmissably, when the tool commands cannot be found by name."""

    resolved = bin_dir.expanduser().resolve()
    if str(resolved) in os.environ.get("PATH", "").split(os.pathsep):
        return
    profile = "~/.zshrc" if os.environ.get("SHELL", "").endswith("zsh") else "~/.bashrc"
    typer.echo(
        f"\n{'=' * 72}\n"
        f"The tool commands will NOT work yet: {resolved} is not on your PATH,\n"
        f"so your shell cannot find {', '.join(wrappers.wrapper_tools())}.\n\n"
        f"Fix it with:\n"
        f'  echo \'export PATH="{resolved}:$PATH"\' >> {profile}\n'
        f"  source {profile}\n"
        f"{'=' * 72}"
    )


@app.command("install-wrappers")
def install_wrappers_command(
    bin_dir: Path = typer.Option(
        wrappers.DEFAULT_WRAPPER_DIR, "--bin-dir", help="Where tool commands are written"
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace files that structbio did not generate"
    ),
) -> None:
    """Write one short shell command per tool, such as 'rfdiffusion'."""

    for path, state in wrappers.install_wrappers(bin_dir, force=force):
        typer.echo(f"{state:<32} {path}")
    _warn_about_path(bin_dir)


@app.command("shell-init")
def shell_init(
    bin_dir: Path = typer.Option(
        wrappers.DEFAULT_WRAPPER_DIR, "--bin-dir", help="Directory holding the tool commands"
    ),
) -> None:
    """Print the PATH line for the tool commands: eval "$(structbio shell-init)"."""

    typer.echo(f'export PATH="{bin_dir.expanduser().resolve()}:$PATH"')


@app.command("config")
def config_command() -> None:
    """Show which configuration files are in use and where tools are installed."""

    settings = load_settings()
    typer.echo(f"{'Lab config':<16} {lab_config_path()}")
    typer.echo(f"{'User config':<16} {user_config_path()}")
    typer.echo(f"{'Experiments':<16} {settings.experiments_root}")
    typer.echo("")
    for name, installation in sorted(settings.tools.items()):
        typer.echo(f"{name}")
        typer.echo(f"{'  path':<16} {installation.path or '(not set)'}")
        typer.echo(f"{'  executable':<16} {installation.executable or '(not set)'}")
        typer.echo(f"{'  manager':<16} {installation.manager}")
        typer.echo(f"{'  environment':<16} {installation.environment or '(none)'}")


@app.command("validate")
def validate(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Validate any structbio YAML configuration."""

    _validate_command(config, set_values=set_value)


@app.command("tools")
def tools_command() -> None:
    """Show implemented backends and other detected scientific tools."""

    settings = load_settings()
    typer.echo(f"{'Tool':<20} {'Status':<16} Backend")
    for backend in get_backends().values():
        installation = settings.tools.get(backend.name)
        check = backend.check_environment(installation) if installation else None
        status = "configured" if check and check.configured else "not configured"
        if check and check.found:
            status = "found"
        typer.echo(f"{backend.display_name:<20} {status:<16} yes")
    for name, path in detect_unwrapped_tools().items():
        if path:
            typer.echo(f"{name:<20} {'detected':<16} no")
    typer.echo("\nQuick commands")
    for line in (
        "rfdiffusion monomer 150 my_monomers",
        "rfdiffusion binder target.pdb 100 my_binders --hotspots B30,B33",
        "proteinmpnn design 7kdp.pdb 8 my_sequences",
        "colabfold predict my_sequences my_folds",
        "cryozeta predict map.mrc chains.fasta my_model --resolution 3.0 --contour 0.3",
    ):
        typer.echo(f"  {line}")


@app.command("doctor")
def doctor() -> None:
    """Diagnose this workstation and the tool installations it can reach."""

    settings = load_settings()
    gpu = detect_gpu()
    typer.echo("structbio workstation\n")
    python_found = shutil.which("python") or shutil.which("python3")
    typer.echo(f"{'Python':<22} {'OK' if python_found else 'NOT FOUND'}")
    typer.echo(f"{'Git':<22} {'OK' if shutil.which('git') else 'NOT FOUND'}")
    typer.echo(f"{'GPU':<22} {'OK' if gpu['available'] else 'NOT FOUND'}")
    for model in gpu["models"]:
        typer.echo(f"{'':<22} {model}")
    if gpu["cuda_driver"]:
        typer.echo(f"{'CUDA driver':<22} {gpu['cuda_driver']}")
    typer.echo("")
    for backend in get_backends().values():
        installation = settings.tools.get(backend.name)
        check = backend.check_environment(installation) if installation else None
        status = "FOUND" if check and check.found else "NOT CONFIGURED"
        if check and check.configured and not check.found:
            status = "CONFIGURED, UNAVAILABLE"
        typer.echo(f"{backend.display_name:<22} {status}")
        if installation and installation.environment:
            typer.echo(f"{'Environment':<22} {installation.environment}")
        if check:
            for detail in check.details:
                typer.echo(f"{'':<22} {detail}")
        typer.echo("")
    typer.echo("Optional detected tools")
    for name, path in detect_unwrapped_tools().items():
        typer.echo(f"{name:<22} {'FOUND' if path else 'NOT FOUND'}")
    slurm = "FOUND" if shutil.which("sbatch") else "NOT FOUND"
    typer.echo(f"{'SLURM (cluster only)':<22} {slurm}")


@app.command("status")
def status(
    target: str | None = typer.Argument(
        None, help="An output folder, or an experiment id below the experiments root"
    ),
    root: Path | None = typer.Option(None, help="Experiments directory"),
) -> None:
    """Show what a run recorded, either for an output folder or an experiment."""

    if target:
        candidate = Path(target).expanduser()
        if candidate.is_dir():
            metadata = read_metadata(candidate)
            if metadata is None:
                _abort(f"No structbio run record found in {candidate}")
            for key in (
                "tool",
                "status",
                "created_at",
                "return_code",
                "command",
                "hostname",
                "structbio_git_commit",
                "wrapped_tool_git_commit",
            ):
                if metadata.get(key) is not None:
                    typer.echo(f"{key:<24} {metadata[key]}")
            return

    experiments_root = (root or load_settings().experiments_root).expanduser().resolve()
    if not experiments_root.is_dir():
        typer.echo(f"No experiments found at {experiments_root}")
        typer.echo("For a quick run, pass its output folder: structbio status my_designs")
        return
    candidates = sorted(path for path in experiments_root.iterdir() if path.is_dir())
    if target:
        candidates = [path for path in candidates if path.name == target]
        if not candidates:
            _abort(f"Experiment not found: {target}")
    typer.echo(f"{'Experiment':<42} {'Tool':<15} Status")
    for candidate in candidates:
        metadata_path = candidate / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            typer.echo(f"{candidate.name:<42} {'?':<15} invalid metadata")
            continue
        status_value = metadata.get("status", "unknown")
        job_id = metadata.get("slurm_job_id")
        if job_id and shutil.which("squeue"):
            result = subprocess.run(
                ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                status_value = f"{status_value} / SLURM {result.stdout.strip()}"
        typer.echo(f"{candidate.name:<42} {metadata.get('tool', '?'):<15} {status_value}")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit")
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


_register_yaml_commands(rfdiffusion_app, "rfdiffusion")
_register_yaml_commands(proteinmpnn_app, "proteinmpnn")
_register_yaml_commands(colabfold_app, "colabfold")
_register_yaml_commands(cryozeta_app, "cryozeta")


@proteinmpnn_app.command("inspect-mask")
def inspect_proteinmpnn_mask(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Print exactly which original PDB residues may mutate."""

    _, backend, config, _ = _validated(config_path, "proteinmpnn", set_value)
    assert isinstance(backend, ProteinMPNNBackend)
    assert isinstance(config, ProteinMPNNConfig)
    for index, inspection in enumerate(backend.inspections(config)):
        if index:
            typer.echo("\n" + "=" * 72 + "\n")
        typer.echo(inspection.render())
    typer.echo("\nMask inversion check: PASSED")
