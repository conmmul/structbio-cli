"""Setting the workstation up: finding software, wiring it, installing it."""

from __future__ import annotations

from pathlib import Path

import typer

from structbio import discovery, install, onboard, wrappers
from structbio.cli.app import app
from structbio.cli.support import _abort
from structbio.config import (
    USER_CONFIG_TEMPLATE,
    lab_config_path,
    load_settings,
    user_config_path,
)


@app.command("setup")
def setup(
    bin_dir: Path = typer.Option(
        wrappers.DEFAULT_WRAPPER_DIR, "--bin-dir", help="Where tool commands are written"
    ),
    detect: bool = typer.Option(True, "--detect/--no-detect", help="Scan for installed software"),
    check: bool = typer.Option(
        True, "--check/--no-check", help="Run each tool's environment check"
    ),
    fix_path: bool = typer.Option(
        True, "--path/--no-path", help="Add the tool commands to PATH in your shell profile"
    ),
    wrappers_only: bool = typer.Option(
        False, "--wrappers-only", help="Do not touch the configuration file"
    ),
) -> None:
    """Set this workstation up completely: find the software, wire it up, check it.

    Safe to run again at any time. It adds what is missing and never overwrites
    an entry you wrote yourself, so it is also the right command after
    installing a new tool.
    """

    config_path = user_config_path()
    found: dict[str, discovery.Discovery] = {}
    if not wrappers_only:
        if detect:
            typer.echo("Scanning for installed software...")
            found = discovery.discover()
            _report_discoveries(found)
            typer.echo("")
        _write_configuration(config_path, found, detect)

    typer.echo("")
    _install_commands(bin_dir)
    path_result = (
        wrappers.ensure_on_path(bin_dir) if fix_path else wrappers.PathResult(
            "on PATH" if wrappers.on_path(bin_dir) else "not on PATH",
            bin_dir.expanduser(),
            wrappers.path_line(bin_dir),
        )
    )
    _report_path(path_result)

    statuses: list[onboard.ToolStatus] = []
    if check and not wrappers_only:
        statuses = _check_environments()
    _report_next_steps(statuses, path_result, found)


def _write_configuration(
    config_path: Path, found: dict[str, discovery.Discovery], detect: bool
) -> None:
    """Create the configuration, or add newly found tools to the existing one."""

    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            discovery.render_config(found) if detect else USER_CONFIG_TEMPLATE,
            encoding="utf-8",
        )
        typer.echo(f"Configuration     {config_path} ({len(found)} tool(s))")
        return
    if found:
        _update_config(config_path, found)
    else:
        typer.echo(f"Configuration     {config_path}")


def _install_commands(bin_dir: Path) -> None:
    results = wrappers.install_wrappers(bin_dir)
    skipped = [(path, state) for path, state in results if state.startswith("skipped")]
    names = ", ".join(sorted(path.name for path, _ in results))
    typer.echo(f"Commands          {bin_dir.expanduser()}  ({names})")
    for path, state in skipped:
        typer.echo(f"                  {state}: {path}")


def _report_path(result: wrappers.PathResult) -> None:
    if result.ready:
        typer.echo(f"PATH              already includes {result.bin_dir}")
        return
    if result.state == "added":
        typer.echo(f"PATH              added to {result.profile}")
        return
    if result.state == "already in profile":
        typer.echo(f"PATH              set in {result.profile}, but this shell has not read it")
        return
    typer.echo(
        f"PATH              could not be set: no shell start-up file you can write.\n"
        f"                  Add this line to your shell configuration yourself:\n"
        f"                    {result.line}"
    )


def _check_environments() -> list[onboard.ToolStatus]:
    """Prove each configured tool can actually run, and adopt what already works."""

    statuses = onboard.review(load_settings())
    if not statuses:
        return []
    typer.echo("\nChecking that each tool can run (this uses the GPU briefly)...")
    for status in statuses:
        typer.echo(f"  {status.tool:<14} {status.state:<16} {status.detail}")
        if status.adopted and status.environment:
            written = onboard.record_environment(status.tool, status.environment)
            if written:
                typer.echo(f"  {'':<14} recorded environment {status.environment} in {written}")
        if status.fix:
            typer.echo(f"  {'':<14} fix: {status.fix}")
    return statuses


def _report_next_steps(
    statuses: list[onboard.ToolStatus],
    path_result: wrappers.PathResult,
    found: dict[str, discovery.Discovery],
) -> None:
    typer.echo("")
    if path_result.needs_reload:
        typer.echo(f"Open a new terminal, or run:  source {path_result.profile}\n")
    ready = [status.tool for status in statuses if status.ready]
    if ready:
        typer.echo("Ready to run: " + ", ".join(ready))
        typer.echo(f"\nTry it:\n  {EXAMPLE_RUNS.get(ready[0], EXAMPLE_RUNS['rfdiffusion'])}")
        return
    if not found:
        typer.echo("No wrapped software was found on this machine. Install one with:")
        typer.echo("  structbio install rfdiffusion --into ~/software")
        return
    typer.echo("Run 'structbio doctor' for the full picture of what is left.")


EXAMPLE_RUNS = {
    "rfdiffusion": "rfdiffusion monomer 100 my_first_designs -n 2",
    "proteinmpnn": "proteinmpnn design my_backbone.pdb 4 my_sequences",
    "colabfold": "colabfold predict my_sequences my_folds --msa-mode single_sequence",
    "cryozeta": "cryozeta predict map.mrc chains.fasta my_model --resolution 3.0 --contour 0.3",
}


def _report_discoveries(found: dict[str, discovery.Discovery]) -> None:
    for signature in discovery.SIGNATURES:
        entry = found.get(signature.tool)
        state = "found" if entry else "not found"
        detail = entry.describe() if entry else ""
        typer.echo(f"  {signature.tool:<14} {state:<10} {detail}".rstrip())


def _update_config(config_path: Path, found: dict[str, discovery.Discovery]) -> None:
    existing = config_path.read_text(encoding="utf-8")
    try:
        merged = discovery.merge_into_config(existing, found)
    except ValueError as exc:
        _abort(str(exc))
    if merged == existing:
        typer.echo(f"Configuration already covers everything found: {config_path}")
        return
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    backup.write_text(existing, encoding="utf-8")
    config_path.write_text(merged, encoding="utf-8")
    typer.echo(f"Updated {config_path} (previous version saved as {backup.name}).")


@app.command("detect")
def detect_command() -> None:
    """Look for wrapped software installed on this machine, changing nothing."""

    found = discovery.discover()
    typer.echo(f"{'Tool':<14} {'Status':<10} Where")
    _report_discoveries(found)
    typer.echo("")
    for name, value in discovery.environment_summary().items():
        typer.echo(f"{name:<14} {value or 'not found'}")
    missing = [s.tool for s in discovery.SIGNATURES if s.tool not in found]
    if missing:
        typer.echo(
            "\nNot found: "
            + ", ".join(missing)
            + "\nInstall one with: structbio install TOOL --into ~/software"
        )
    elif found:
        typer.echo("\nRecord these in the configuration with: structbio setup")


@app.command("install")
def install_command(
    tool: str = typer.Argument(..., help="rfdiffusion, proteinmpnn, colabfold, or cryozeta"),
    into: Path = typer.Option(
        Path("~/software"), "--into", help="Directory to clone the project into"
    ),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before cloning"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan and clone nothing"),
    configure: bool = typer.Option(
        True, "--configure/--no-configure", help="Record the path in the configuration"
    ),
) -> None:
    """Clone a wrapped project, then print its own remaining setup steps.

    structbio does not create environments or download model weights: those
    differ per machine, change with each release, and in CryoZeta's case carry
    a licence only you can accept.
    """

    try:
        recipe = install.recipe_for(tool)
    except ValueError as exc:
        _abort(str(exc))
    target = recipe.target(into)

    typer.echo(f"{recipe.display_name}")
    typer.echo(f"{'  repository':<16} {recipe.repository}")
    typer.echo(f"{'  clone into':<16} {target}")
    typer.echo(f"{'  licence':<16} {recipe.licence}")
    typer.echo(f"{'  weights':<16} {recipe.weights}")
    typer.echo(f"{'  verified':<16} upstream README read on {recipe.verified_on}")

    if dry_run:
        typer.echo("\nDry run: nothing was cloned.")
        _show_remaining_steps(recipe, target)
        return
    if target.exists():
        _abort(f"{target} already exists; remove it, or point --into somewhere else")
    if not yes and not typer.confirm(f"\nClone {recipe.repository} into {target}?"):
        typer.echo("Nothing was cloned.")
        raise typer.Exit(1)

    try:
        install.clone(recipe, into)
    except (FileExistsError, RuntimeError) as exc:
        _abort(str(exc))
    typer.echo(f"\nCloned into {target}")

    if configure:
        found = {
            name: entry
            for name, entry in discovery.discover(roots=(str(into.expanduser()),)).items()
            if name == recipe.tool
        }
        config_path = user_config_path()
        if not found:
            typer.echo("The checkout is not usable yet, so nothing was configured.")
        elif config_path.exists():
            _update_config(config_path, found)
        else:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(discovery.render_config(found), encoding="utf-8")
            typer.echo(f"Wrote {config_path}")
    _show_remaining_steps(recipe, target)


def _show_remaining_steps(recipe: install.InstallRecipe, target: Path) -> None:
    typer.echo(f"\nRemaining steps, from the {recipe.display_name} README:\n")
    for step in recipe.rendered_steps(target):
        typer.echo(f"  {step}")
    for note in recipe.notes:
        typer.echo(f"\n{note.format(directory=target)}")
    typer.echo("\nThen: structbio setup && structbio doctor")


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
    _report_path(wrappers.ensure_on_path(bin_dir))


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
