"""Typer command-line interface for structbio."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from structbio import __version__
from structbio.config import LoadedConfig, load_config, load_settings, parse_cli_overrides
from structbio.environment import detect_gpu, detect_unwrapped_tools
from structbio.execution import execute_plan, plan_input_paths
from structbio.experiment import ExperimentManager, update_metadata, write_records
from structbio.slurm import generate_slurm_script
from structbio.tools import get_backend, get_backends
from structbio.tools.base import BackendContext, CommandPlan, ToolBackend, ValidationReport
from structbio.tools.proteinmpnn import ProteinMPNNBackend, ProteinMPNNConfig


app = typer.Typer(
    name="structbio",
    help="Safe, reproducible structural-biology tool orchestration.",
    no_args_is_help=True,
)
rfdiffusion_app = typer.Typer(help="RFdiffusion workflows", no_args_is_help=True)
proteinmpnn_app = typer.Typer(help="ProteinMPNN sequence design", no_args_is_help=True)
cryozeta_app = typer.Typer(help="CryoZeta inference", no_args_is_help=True)
app.add_typer(rfdiffusion_app, name="rfdiffusion")
app.add_typer(proteinmpnn_app, name="proteinmpnn")
app.add_typer(cryozeta_app, name="cryozeta")


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
        tool = loaded.data.get("tool")
        if not isinstance(tool, str):
            _abort("Configuration must contain a string 'tool' field")
        if expected_tool and tool != expected_tool:
            _abort(f"Expected tool: {expected_tool}; configuration selects: {tool}")
        backend = get_backend(tool)
        config = backend.parse_config(loaded.data, loaded.source)
        return loaded, backend, config
    except (ValueError, ValidationError) as exc:
        _abort(str(exc))
    raise AssertionError("unreachable")


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


def _context(
    loaded: LoadedConfig, backend: ToolBackend, config: Any, experiment_dir: Path
) -> BackendContext:
    installation = loaded.settings.tools.get(backend.name)
    if installation is None:
        _abort(f"No installation settings exist for {backend.display_name}")
    return BackendContext(
        source=loaded.source,
        installation=installation,
        experiment_dir=experiment_dir,
        output_dir=experiment_dir / "outputs",
        inputs_dir=experiment_dir / "inputs",
    )


def _preview_plan(
    loaded: LoadedConfig, backend: ToolBackend, config: Any
) -> tuple[Path, CommandPlan]:
    experiment_dir = _manager(loaded).candidate(config.experiment.name)
    context = _context(loaded, backend, config, experiment_dir)
    return experiment_dir, backend.build_command(config, context)


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
    config_path: Path, tool: str, dry_run: bool, set_values: list[str] | None = None
) -> None:
    loaded, backend, config, report = _validated(config_path, tool, set_values)
    if dry_run:
        experiment_dir, plan = _preview_plan(loaded, backend, config)
        _show_report(report)
        typer.echo(f"\nCommand:\n{plan.render()}")
        typer.echo(f"\nExperiment: {experiment_dir}")
        typer.echo(f"Output: {plan.output_dir}")
        typer.echo("Dry run: nothing was created or executed.")
        return

    installation = loaded.settings.tools[backend.name]
    environment = backend.check_environment(installation)
    if not environment.found:
        details = "; ".join(environment.details)
        _abort(f"{backend.display_name} installation is unavailable. {details}")
    paths = _manager(loaded).create(config.experiment.name)
    context = _context(loaded, backend, config, paths.root)
    plan = backend.build_command(config, context)
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
    return_code = execute_plan(plan, paths)
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
        plan = backend.build_command(config, _context(loaded, backend, config, paths.root))

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
    installation = loaded.settings.tools[backend.name]
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


@app.command("validate")
def validate(
    config: Path = typer.Argument(..., exists=True, dir_okay=False),
    set_value: list[str] = typer.Option([], "--set", help="Override dotted.key=YAML_VALUE"),
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


@app.command("doctor")
def doctor() -> None:
    """Diagnose core commands and configured tool installations."""

    settings = load_settings()
    gpu = detect_gpu()
    checks = {
        "Python": shutil.which("python") or shutil.which("python3"),
        "Git": shutil.which("git"),
        "SLURM": shutil.which("sbatch") or shutil.which("squeue"),
        "CUDA": "available" if gpu["available"] else None,
    }
    typer.echo("structbio environment\n")
    for name, value in checks.items():
        typer.echo(f"{name:<22} {'OK' if value else 'NOT FOUND'}")
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
    detected = detect_unwrapped_tools()
    for name, path in detected.items():
        typer.echo(f"{name:<22} {'FOUND' if path else 'NOT FOUND'}")


@app.command("status")
def status(
    experiment: str | None = typer.Argument(None),
    root: Path | None = typer.Option(None, help="Experiments directory"),
) -> None:
    """Inspect recorded experiments and their SLURM jobs."""

    experiments_root = (root or load_settings().experiments_root).expanduser().resolve()
    if not experiments_root.is_dir():
        typer.echo(f"No experiments found at {experiments_root}")
        return
    candidates = sorted(path for path in experiments_root.iterdir() if path.is_dir())
    if experiment:
        candidates = [path for path in candidates if path.name == experiment]
        if not candidates:
            _abort(f"Experiment not found: {experiment}")
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
        typer.echo(
            f"{candidate.name:<42} {metadata.get('tool', '?'):<15} {status_value}"
        )


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit")
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


def _register_standard_commands(subapp: typer.Typer, tool: str) -> None:
    @subapp.command("validate")
    def validate_tool(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        set_value: list[str] = typer.Option([], "--set", help="Override dotted.key=YAML_VALUE"),
    ) -> None:
        _validate_command(config, tool, set_value)

    @subapp.command("command")
    def command_tool(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        set_value: list[str] = typer.Option([], "--set", help="Override dotted.key=YAML_VALUE"),
    ) -> None:
        _command_command(config, tool, set_value)

    @subapp.command("run")
    def run_tool(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        dry_run: bool = typer.Option(False, "--dry-run", help="Validate and plan only"),
        set_value: list[str] = typer.Option([], "--set", help="Override dotted.key=YAML_VALUE"),
    ) -> None:
        _run_command(config, tool, dry_run, set_value)

    @subapp.command("submit")
    def submit_tool(
        config: Path = typer.Argument(..., exists=True, dir_okay=False),
        dry_run: bool = typer.Option(False, "--dry-run", help="Print script only"),
        execute: bool = typer.Option(False, "--execute", help="Explicitly invoke sbatch"),
        set_value: list[str] = typer.Option([], "--set", help="Override dotted.key=YAML_VALUE"),
    ) -> None:
        _submit_command(config, tool, dry_run, execute, set_value)


_register_standard_commands(rfdiffusion_app, "rfdiffusion")
_register_standard_commands(proteinmpnn_app, "proteinmpnn")
_register_standard_commands(cryozeta_app, "cryozeta")


@proteinmpnn_app.command("inspect-mask")
def inspect_proteinmpnn_mask(
    config_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    set_value: list[str] = typer.Option([], "--set", help="Override dotted.key=YAML_VALUE"),
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
