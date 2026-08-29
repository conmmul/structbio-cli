"""Configure a tool the first time it is needed, instead of before it is.

`structbio setup` writes the configuration up front, but a researcher who has
just cloned this repository should not have to know that. When a command needs
a tool that no configuration mentions, structbio looks for it in the same
places `setup` looks, uses what it finds, and records it so the next run does
not have to look again.

Only paths to software that is already installed are written here. Nothing is
created, downloaded, or accepted on the researcher's behalf.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from structbio import discovery
from structbio.config import ToolInstallation, user_config_path


DISABLE_VARIABLE = "STRUCTBIO_NO_AUTOCONFIG"


@dataclass(frozen=True)
class Adoption:
    """A tool found on this machine, and where that fact was recorded."""

    tool: str
    found: discovery.Discovery
    installation: ToolInstallation
    config_path: Path | None
    note: str | None = None

    def describe(self) -> str:
        return f"Found {self.tool} at {self.found.describe()}"


def enabled() -> bool:
    """False when the researcher has asked for explicit configuration only."""

    return os.environ.get(DISABLE_VARIABLE, "").strip().lower() not in ("1", "true", "yes")


def find(tool: str) -> discovery.Discovery | None:
    """Look for one tool, without scanning for the others."""

    signatures = tuple(item for item in discovery.SIGNATURES if item.tool == tool)
    if not signatures:
        return None
    return discovery.discover(signatures).get(tool)


def record(tool: str, found: discovery.Discovery, config_path: Path | None = None) -> Path | None:
    """Add a discovered tool to the user configuration, changing nothing else.

    Returns the file written, or None when it could not be written: an
    unwritable configuration is a reason to skip the bookkeeping, never a
    reason to refuse the run.
    """

    path = config_path or user_config_path()
    try:
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            merged = discovery.merge_into_config(existing, {tool: found})
            if merged == existing:
                return None
            path.with_suffix(path.suffix + ".bak").write_text(existing, encoding="utf-8")
        else:
            merged = discovery.render_config({tool: found})
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(merged, encoding="utf-8")
    except (OSError, ValueError):
        return None
    return path


def adopt(tool: str, *, config_path: Path | None = None) -> Adoption | None:
    """Find a tool that is not configured yet, and remember where it is."""

    if not enabled():
        return None
    found = find(tool)
    if found is None:
        return None
    installation = ToolInstallation.model_validate(found.settings())
    written = record(tool, found, config_path)
    note = None if written else "Could not update the configuration; using it for this run only."
    return Adoption(
        tool=tool,
        found=found,
        installation=installation,
        config_path=written,
        note=note,
    )
