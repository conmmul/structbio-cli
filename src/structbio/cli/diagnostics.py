"""Diagnostics: what this machine can reach, and what a run recorded."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer

from structbio import __version__, environment
from structbio.cli.app import SET_HELP, app
from structbio.cli.configs import _validate_command
from structbio.cli.support import _abort
from structbio.config import load_settings
from structbio.environment import detect_unwrapped_tools
from structbio.experiment import direct_paths, read_metadata, tail
from structbio.tools import get_backends


@app.command("gpu")
def gpu_command() -> None:
    """Show what this machine reports about its GPUs, and why if it cannot."""

    report = environment.gpu_report()
    typer.echo(f"{'nvidia-smi':<22} {report.executable or 'NOT FOUND'}")
    if report.driver_version:
        typer.echo(f"{'driver':<22} {report.driver_version}")
    if report.driver_cuda:
        typer.echo(f"{'driver CUDA':<22} {report.driver_cuda}")

    if report.error:
        typer.echo(f"\nNo GPU information: {report.error}\n")
        typer.echo("Things to check, in order:")
        typer.echo("  nvidia-smi                     does it work outside structbio?")
        typer.echo("  command -v nvidia-smi          is it on PATH in this shell?")
        typer.echo("  STRUCTBIO_NVIDIA_SMI=/path/to/nvidia-smi structbio gpu")
        typer.echo("  ls /usr/bin/nvidia-smi         is the driver installed at all?")
        typer.echo(
            "\nIf nvidia-smi works in your own shell but not here, structbio is being "
            "run somewhere with a different PATH, such as a container or a cron job."
        )
        raise typer.Exit(1)

    typer.echo("")
    for index, name in enumerate(report.names):
        capability = (
            report.capabilities[index]
            if index < len(report.capabilities)
            else None
        )
        described = f"{capability[0]}.{capability[1]}" if capability else "unknown"
        typer.echo(f"  GPU {index}: {name}  (compute capability {described})")

    if report.capability_source == "model name":
        typer.echo(
            "\nThe driver did not report compute capability, so it was inferred "
            "from the model name. Update the driver for a definitive answer."
        )
    elif report.capability_source == "none":
        typer.echo(
            "\nCompute capability is unknown: the driver does not report it and the "
            "model name is not recognised. Pass it yourself where a command needs "
            "it, for example: structbio env create rfdiffusion --capability 8.9"
        )


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
    gpu = environment.detect_gpu()
    typer.echo("structbio workstation\n")
    # Name the running installation: several checkouts on one machine is the
    # usual reason a command is reported as missing.
    typer.echo(f"{'structbio':<22} {__version__}")
    typer.echo(f"{'  package':<22} {Path(__file__).resolve().parent}")
    typer.echo(f"{'  interpreter':<22} {sys.executable}")
    python_found = shutil.which("python") or shutil.which("python3")
    typer.echo(f"{'Python':<22} {'OK' if python_found else 'NOT FOUND'}")
    typer.echo(f"{'Git':<22} {'OK' if shutil.which('git') else 'NOT FOUND'}")
    typer.echo(f"{'GPU':<22} {'OK' if gpu['available'] else 'NOT FOUND'}")
    for model in gpu["models"]:
        typer.echo(f"{'':<22} {model}")
    if not gpu["available"] and gpu.get("error"):
        typer.echo(f"{'':<22} {gpu['error']}")
        typer.echo(f"{'':<22} more detail: structbio gpu")
    if gpu["cuda_driver"]:
        typer.echo(f"{'CUDA driver':<22} {gpu['cuda_driver']}")
    typer.echo("")
    for backend in get_backends().values():
        installation = settings.tools.get(backend.name)
        check = backend.check_environment(installation) if installation else None
        status = "FOUND" if check and check.found else "NOT CONFIGURED"
        if check and check.found and check.warnings:
            status = "FOUND, WITH WARNINGS"
        elif check and check.configured and not check.found:
            status = "CONFIGURED, UNAVAILABLE"
        typer.echo(f"{backend.display_name:<22} {status}")
        if check:
            for detail in check.details:
                typer.echo(f"{'':<22} {detail}")
            for problem in check.problems:
                typer.echo(f"{'':<22} {problem}")
            for warning in check.warnings:
                typer.echo(f"{'  warning:':<22} {warning}")
            for remedy in check.remedies:
                typer.echo(f"{'  fix:':<22} {remedy}")
        elif installation is None:
            typer.echo(f"{'  fix:':<22} structbio install {backend.name} --into ~/software")
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
            if metadata.get("status") == "failed":
                lines = tail(direct_paths(candidate).stderr)
                if lines:
                    typer.echo("\nLast lines of stderr.log:")
                    for line in lines:
                        typer.echo(f"  {line}")
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

