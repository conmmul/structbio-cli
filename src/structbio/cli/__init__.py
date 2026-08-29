"""Typer command-line interface for structbio.

The command tree lives in `app`; each module below registers its own commands
on import, so importing this package assembles the whole interface.
"""

# ruff: noqa: I001  the command modules are imported in the order they are listed

from __future__ import annotations

import subprocess  # noqa: F401  (tests replace this module's subprocess)

import typer

from structbio import __version__, autoconfig, discovery, environment  # noqa: F401

# Imported for their side effect of registering commands. The order below is
# the order they are listed in `structbio --help`, so setting the workstation
# up comes before diagnosing it; it is deliberately not alphabetical.
from structbio.cli import workstation  # noqa: F401
from structbio.cli import environments  # noqa: F401
from structbio.cli import diagnostics  # noqa: F401
from structbio.cli import runs  # noqa: F401
from structbio.cli import configs  # noqa: F401
from structbio.cli.app import (
    app,
    colabfold_app,
    cryozeta_app,
    proteinmpnn_app,
    rfdiffusion_app,
)
from structbio.cli.environments import env_app


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="Show version and exit")
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit()


__all__ = [
    "app",
    "colabfold_app",
    "cryozeta_app",
    "env_app",
    "proteinmpnn_app",
    "rfdiffusion_app",
]
