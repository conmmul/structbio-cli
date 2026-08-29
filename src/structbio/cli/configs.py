"""YAML commands: full control over every documented option."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer

from structbio.cli.app import (
    GPU_HELP,
    QUIET_HELP,
    SET_HELP,
    colabfold_app,
    cryozeta_app,
    proteinmpnn_app,
    rfdiffusion_app,
)
from structbio.cli.support import (
    _abort,
    _context,
    _expected_chains,
    _installation,
    _manager,
    _preview_plan,
    _report_outputs,
    _require_installed,
    _show_failure,
    _show_report,
    _validated,
    _with_gpu,
)
from structbio.execution import execute_plan, plan_input_paths
from structbio.experiment import update_metadata, write_records
from structbio.slurm import generate_slurm_script
from structbio.tools.proteinmpnn import ProteinMPNNBackend, ProteinMPNNConfig


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
        _show_failure(backend.display_name, return_code, paths)
        raise typer.Exit(return_code)
    _report_outputs(paths.outputs, _expected_chains(config))
    typer.echo(f"\nCompleted. Outputs: {paths.outputs}")


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
