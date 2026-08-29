"""Building and checking the environments the tools run in."""

from __future__ import annotations

import subprocess
from typing import Any

import typer

from structbio import provision
from structbio.cli.app import app
from structbio.cli.support import _abort
from structbio.config import load_settings, user_config_path
from structbio.tools import get_backends


env_app = typer.Typer(
    help=(
        "Build and check the environments the tools run in.\n\n"
        "  structbio env create rfdiffusion\n"
        "  structbio env verify rfdiffusion"
    ),
    no_args_is_help=True,
)
app.add_typer(env_app, name="env")


def _installation_for(tool: str) -> tuple[Any, str]:
    settings = load_settings()
    backends = get_backends()
    if tool not in backends:
        _abort(f"Unknown tool {tool!r}; known tools: {', '.join(sorted(backends))}")
    installation = settings.tools.get(tool)
    if installation is None:
        _abort(f"No installation is configured for {tool}; run 'structbio setup'")
    return installation, installation.environment or ""


def _show_plan(plan: provision.EnvironmentPlan) -> None:
    for note in plan.notes:
        typer.echo(f"  {note}")
    if plan.notes:
        typer.echo("")
    if not plan.upstream_verified:
        typer.echo("  These versions are not an upstream-published combination.\n")
    for index, step in enumerate(plan.steps, start=1):
        typer.echo(f"  {index}. {step.description}")
        typer.echo(f"     {step.render()}")


def _report_probe(tool: str, name: str, result: provision.ProbeResult) -> bool:
    failures = result.failures()
    if result.ok:
        typer.echo(f"  {result.summary()}")
    for failure in failures:
        typer.echo(f"  FAILED: {failure}")
    if not failures:
        typer.echo(f"  {name} is ready for {tool}.")
        return True
    return False


@env_app.command("verify")
def env_verify(
    tool: str = typer.Argument(
        ..., help="rfdiffusion, proteinmpnn or cryozeta"
    ),
) -> None:
    """Run code in the tool's environment to prove it actually works.

    This allocates memory on the GPU, which is what catches a PyTorch that was
    built without support for this particular card: importing it and asking
    whether CUDA is available both succeed in that case.
    """

    installation, name = _installation_for(tool)
    interpreter = provision.tool_interpreter(installation, name)
    if interpreter is None:
        if installation.manager == "pixi":
            _abort(
                f"No pixi environment for {tool} at "
                f"{installation.path}/.pixi/envs/{name or 'default'}; run "
                "'pixi run setup' in that checkout"
            )
        if not name:
            _abort(f"No conda environment is configured for {tool}")
        _abort(
            f"The conda environment {name!r} does not exist; "
            f"run 'structbio env create {tool}'"
        )
    typer.echo(f"Checking {name or 'the environment'}...")
    typer.echo(f"  interpreter {interpreter}")
    if not _report_probe(
        tool, name or str(interpreter), provision.verify(tool, name, interpreter=interpreter)
    ):
        if installation.manager == "pixi":
            # structbio does not build these; the project's own task does.
            typer.echo(
                f"\nThis environment is managed by pixi. Repair it in the checkout:"
                f"\n  cd {installation.path}\n  pixi run setup"
            )
        else:
            typer.echo(f"\nRepair it with: structbio env repair {tool}")
        raise typer.Exit(1)


@env_app.command("adopt")
def env_adopt(
    tool: str = typer.Argument(..., help="rfdiffusion, proteinmpnn, colabfold or cryozeta"),
    environment_name: str = typer.Option(
        ..., "--environment", "-e", help="The conda environment that already works"
    ),
) -> None:
    """Record an environment that already works, instead of building one.

    If you have a working setup, this is the right command: it checks that
    environment by running code in it, and records it if that succeeds. Nothing
    is installed, changed or removed.
    """

    _installation_for(tool)  # rejects an unknown or unconfigured tool
    if not provision.environment_exists(environment_name):
        _abort(f"No conda environment named {environment_name!r}")
    facts = provision.environment_facts(environment_name)
    if facts.get("python"):
        typer.echo(f"Python {facts['python']} at {facts.get('executable', '?')}")
    typer.echo(f"Checking {environment_name} for {tool}...")
    result = provision.verify(tool, environment_name)
    if not _report_probe(tool, environment_name, result):
        typer.echo(
            "\nNot recorded, because it did not pass. If this environment does work "
            "for you, the check is wrong and worth reporting.",
            err=True,
        )
        raise typer.Exit(1)

    config_path = user_config_path()
    if not config_path.exists():
        _abort(f"No configuration at {config_path}; run 'structbio setup' first")
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tools = data.setdefault("tools", {})
    entry = tools.setdefault(tool, {})
    entry["environment"] = environment_name
    entry.setdefault("manager", "conda")
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    typer.echo(
        f"\nRecorded {environment_name} for {tool} in {config_path} "
        f"(previous version saved as {backup.name})."
    )


@env_app.command("repair")
def env_repair(
    tool: str = typer.Argument(..., help="rfdiffusion or proteinmpnn"),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before installing"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan and change nothing"),
) -> None:
    """Put a CUDA-capable PyTorch into the environment you already have.

    Nothing is removed and nothing else in the environment changes. Use this
    rather than 'env create' whenever the environment is otherwise working.
    """

    installation, _ = _installation_for(tool)
    plan = provision.repair_plan(tool, installation)
    typer.echo(f"{tool}: repairing {plan.environment}\n")
    if plan.blocked:
        typer.echo(f"Cannot repair this environment: {plan.blocked}\n")
        for alternative in plan.alternatives:
            typer.echo(f"  {alternative}")
        raise typer.Exit(1)

    _show_plan(plan)
    if dry_run:
        typer.echo("\nDry run: nothing was changed.")
        return
    if not yes and not typer.confirm(
        f"\nChange PyTorch in {plan.environment}? Its current PyTorch is replaced"
    ):
        typer.echo("Nothing was changed.")
        raise typer.Exit(1)

    for index, step in enumerate(plan.steps, start=1):
        typer.echo(f"\n[{index}/{len(plan.steps)}] {step.description}")
        result = subprocess.run(list(step.argv), check=False)
        if result.returncode:
            typer.echo(
                f"\nStep {index} failed with exit code {result.returncode}:\n"
                f"  {step.render()}\nThe environment is unchanged from this step "
                "onwards; earlier steps did apply.",
                err=True,
            )
            raise typer.Exit(result.returncode)

    typer.echo("\nChecking that the GPU is now usable...")
    if not _report_probe(tool, plan.environment, provision.verify(tool, plan.environment)):
        raise typer.Exit(1)


@env_app.command("create")
def env_create(
    tool: str = typer.Argument(..., help="rfdiffusion or proteinmpnn"),
    capability: str | None = typer.Option(
        None,
        "--capability",
        help="Compute capability such as 8.9, when the driver cannot report it",
    ),
    force: bool = typer.Option(False, "--force", help="Replace an existing environment"),
    yes: bool = typer.Option(False, "--yes", help="Do not ask before building"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the plan and build nothing"),
) -> None:
    """Build the environment this machine needs, then prove it works.

    The versions are chosen from the GPU actually present. Where no working
    combination exists, this says so rather than building something that cannot
    run.
    """

    installation, configured = _installation_for(tool)
    stated: tuple[int, int] | None = None
    if capability:
        try:
            major, minor = (int(part) for part in capability.split(".", 1))
            stated = (major, minor)
        except ValueError:
            _abort(f"Invalid --capability {capability!r}; write it as 8.9")
    plan = provision.plan_environment(tool, installation, capability=stated)
    typer.echo(f"{tool}: environment {plan.environment}\n")

    if plan.blocked:
        typer.echo(f"Cannot build this environment: {plan.blocked}\n")
        if plan.alternatives:
            typer.echo("What you can do instead:")
            for alternative in plan.alternatives:
                typer.echo(f"  {alternative}")
        raise typer.Exit(1)

    _show_plan(plan)
    if dry_run:
        typer.echo("\nDry run: nothing was built.")
        return

    exists = provision.environment_exists(plan.environment)
    if exists:
        working = provision.verify(tool, plan.environment)
        if not working.failures():
            typer.echo(
                f"\n{plan.environment} already works: {working.summary()}\n"
                "Nothing was built. Record it with: "
                f"structbio env adopt {tool} --environment {plan.environment}"
            )
            raise typer.Exit(0)
    if exists and not force:
        typer.echo(
            f"\n{plan.environment} already exists. Check it with "
            f"'structbio env verify {tool}', or rebuild it with --force."
        )
        raise typer.Exit(1)
    if not yes and not typer.confirm(
        f"\nBuild {plan.environment} now? This downloads several GB"
    ):
        typer.echo("Nothing was built.")
        raise typer.Exit(1)
    moved: str | None = None
    if exists:
        typer.echo(f"Moving the existing {plan.environment} aside...")
        moved = provision.move_aside(plan.environment)
        if moved is None or provision.environment_exists(plan.environment):
            _abort(
                f"{plan.environment} could not be renamed, so rebuilding would land "
                "on top of it. Move it aside yourself with "
                f"'conda rename -n {plan.environment} {plan.environment}-old' and "
                "try again. Nothing was changed."
            )
        typer.echo(
            f"  kept as {moved}; restore it with "
            f"'conda rename -n {moved} {plan.environment}'"
        )

    for index, step in enumerate(plan.steps, start=1):
        typer.echo(f"\n[{index}/{len(plan.steps)}] {step.description}")
        captured: list[str] = []
        process = subprocess.Popen(
            list(step.argv),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            captured.append(line)
            typer.echo(line.rstrip())
        returncode = process.wait()
        if not returncode and index == 1:
            mismatch = provision.unusable_python(plan.environment)
            if mismatch:
                typer.echo("\n" + mismatch, err=True)
                raise typer.Exit(1)
            facts = provision.environment_facts(plan.environment)
            if facts.get("python"):
                typer.echo(
                    f"     Python {facts['python']} at {facts.get('executable', '?')}"
                )
        if returncode:
            typer.echo(
                f"\nStep {index} failed with exit code {returncode}:\n"
                f"  {step.render()}",
                err=True,
            )
            for explanation in provision.explain_pip_failure(
                plan.environment, "".join(captured)
            ):
                typer.echo(f"\n{explanation}", err=True)
            if moved:
                typer.echo(
                    f"\nYour previous environment is intact as {moved}. Restore it "
                    f"with: conda rename -n {moved} {plan.environment}",
                    err=True,
                )
            raise typer.Exit(returncode)

    typer.echo("\nBuilt. Checking that it really works...")
    if not _report_probe(tool, plan.environment, provision.verify(tool, plan.environment)):
        typer.echo(
            "\nThe environment built but does not work. Nothing was removed, so "
            "the output above can be sent to whoever supports this workstation.",
            err=True,
        )
        raise typer.Exit(1)
    if configured != plan.environment:
        typer.echo(
            f"\nRecord it with: structbio setup, or set "
            f"'environment: {plan.environment}' for {tool} in the configuration."
        )

