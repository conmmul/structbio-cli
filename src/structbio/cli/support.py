"""Helpers shared by the command modules."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from structbio import autoconfig, environment, quick
from structbio.config import (
    LoadedConfig,
    load_config,
    parse_cli_overrides,
    user_config_path,
)
from structbio.experiment import ExperimentManager, ExperimentPaths, tail
from structbio.tools import get_backend
from structbio.tools.base import BackendContext, CommandPlan, ToolBackend, ValidationReport
from structbio.validation import StructureValidationError, parse_pdb


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


MODEL_SUFFIXES = (".pdb", ".cif", ".mmcif", ".ent")


def _chain_summary(path: Path) -> str | None:
    """Chains and residue counts in a model file, or None if unreadable."""

    if path.suffix.lower() not in (".pdb", ".ent"):
        return None
    try:
        structure = parse_pdb(path)
    except StructureValidationError:
        return None
    chains = sorted(structure.chains)
    shown = ", ".join(f"{name}:{len(structure.for_chain(name))}" for name in chains[:10])
    more = "" if len(chains) <= 10 else f", +{len(chains) - 10} more"
    return f"{len(chains)} chains ({shown}{more})"


def _report_outputs(output_dir: Path, expected_chains: int | None = None) -> None:
    """Name the model files a run produced, and what is in them.

    A tool can write detection output, intermediates and a final model into one
    folder, and opening the wrong one looks exactly like a broken prediction.
    """

    models = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES
    )
    if not models:
        return
    typer.echo(f"\nModel files ({len(models)}):")
    mismatched = False
    for path in models[:15]:
        summary = _chain_summary(path)
        line = str(path.relative_to(output_dir))
        typer.echo(f"  {line}" + (f"   {summary}" if summary else ""))
        if expected_chains and summary:
            found = int(summary.split(" ", 1)[0])
            if found != expected_chains:
                mismatched = True
    if len(models) > 15:
        typer.echo(f"  ... and {len(models) - 15} more")
    if expected_chains:
        typer.echo(f"\nChains requested: {expected_chains}")
        if mismatched:
            typer.echo(
                "  Some files hold a different number of chains. A pipeline that "
                "writes detection output and intermediates alongside its final "
                "model will do this; check which file is the final one before "
                "judging the prediction."
            )


def _show_failure(display_name: str, return_code: int, paths: ExperimentPaths) -> None:
    """Put the tool's own last words next to the failure, not in a file."""

    typer.echo(f"\n{display_name} failed with exit code {return_code}.", err=True)
    for stream, path in (("stderr", paths.stderr), ("stdout", paths.stdout)):
        lines = tail(path)
        if lines:
            typer.echo(f"\nLast lines of {stream}.log:", err=True)
            for line in lines:
                typer.echo(f"  {line}", err=True)
            break
    typer.echo(f"\nFull logs: {paths.stderr.parent}", err=True)


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


def _expected_chains(config: Any) -> int | None:
    """How many chains the request describes, when the tool states it."""

    try:
        from structbio.tools.cryozeta import CryoZetaConfig, target_chains

        if isinstance(config, CryoZetaConfig):
            return target_chains(config)
    except (ValueError, OSError):
        return None
    return None


def _manager(loaded: LoadedConfig) -> ExperimentManager:
    root = loaded.settings.experiments_root
    if not root.is_absolute():
        root = Path.cwd() / root
    return ExperimentManager(root)


def _installation(loaded: LoadedConfig, backend: ToolBackend) -> Any:
    """Where this tool lives, looking for it on disk if nothing says.

    A researcher who has just installed structbio has no configuration yet.
    Rather than sending them to 'structbio setup' first, the same scan setup
    would run happens here, and what it finds is recorded for next time.
    """

    installation = loaded.settings.tools.get(backend.name)
    if installation is not None and backend.check_environment(installation).configured:
        return installation
    adoption = autoconfig.adopt(backend.name)
    if adoption is not None:
        found = adoption.installation
        if installation is not None and installation.environment and not found.environment:
            found.environment = installation.environment
        typer.echo(adoption.describe())
        if adoption.config_path:
            typer.echo(f"Recorded it in {adoption.config_path}")
        elif adoption.note:
            typer.echo(adoption.note)
        loaded.settings.tools[backend.name] = found
        return found
    if installation is not None:
        # Let the environment check explain precisely what is wrong with it.
        return installation
    _abort(
        f"{backend.display_name} is not installed anywhere structbio looked. "
        f"Install it with 'structbio install {backend.name} --into ~/software', "
        f"or name an existing checkout in {user_config_path()}"
    )


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
    if environment.found:
        for warning in environment.warnings:
            typer.echo(f"WARNING: {warning}")
        if environment.warnings and environment.remedies:
            for remedy in environment.remedies:
                typer.echo(f"         {remedy}")
        return installation
    lines = [f"{backend.display_name} is not available on this machine."]
    lines.extend(f"  {problem}" for problem in environment.problems or ("reason unknown",))
    if environment.remedies:
        lines.append("")
        lines.append("To fix it:")
        lines.extend(f"  {remedy}" for remedy in environment.remedies)
    if backend.needs_torch and installation.environment:
        lines.append("")
        lines.append(
            f"  structbio env create {backend.name}   builds a working environment "
            "for this machine, or explains why none exists"
        )
    typer.echo("Error: " + "\n".join(lines), err=True)
    raise typer.Exit(2)


def _resolve_gpu(gpu: str | None) -> str | None:
    """Turn --gpu into a concrete id list, resolving 'auto' against this machine."""

    if not gpu:
        return None
    if gpu.strip().lower() == "auto":
        index = environment.select_idle_gpu()
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

