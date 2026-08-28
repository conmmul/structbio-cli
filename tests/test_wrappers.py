import os
import subprocess
from pathlib import Path

from structbio import wrappers
from structbio.tools import get_backends


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_repository_wrappers_match_the_generated_template() -> None:
    for tool in wrappers.wrapper_tools():
        path = REPOSITORY_ROOT / "bin" / tool
        assert path.is_file(), f"missing wrapper for {tool}"
        assert path.read_text(encoding="utf-8") == wrappers.render_wrapper(tool)


def test_every_backend_has_a_wrapper_and_structbio_wraps_itself() -> None:
    assert wrappers.wrapper_tools() == [wrappers.SELF_NAME, *sorted(get_backends())]


def test_the_self_wrapper_refuses_to_call_itself(tmp_path: Path) -> None:
    """Without this guard, a wrapper on PATH ahead of the real command loops."""

    directory = tmp_path / "bin"
    wrappers.install_wrappers(directory, launch="structbio")
    script = directory / wrappers.SELF_NAME
    result = subprocess.run(
        ["bash", str(script), "--version"],
        env={**os.environ, "PATH": f"{directory}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 127
    assert "would loop forever" in result.stderr


def test_the_self_wrapper_passes_arguments_straight_through(tmp_path: Path) -> None:
    directory = tmp_path / "bin"
    launcher = tmp_path / "fake-structbio"
    launcher.write_text('#!/usr/bin/env bash\necho "got: $*"\n')
    launcher.chmod(0o755)
    wrappers.install_wrappers(directory, launch=str(launcher))
    result = subprocess.run(
        ["bash", str(directory / wrappers.SELF_NAME), "detect", "--x"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip() == "got: detect --x"


def test_install_writes_executable_wrappers(tmp_path: Path) -> None:
    results = wrappers.install_wrappers(tmp_path / "bin", launch="/opt/venv/bin/structbio")
    assert [state for _, state in results] == ["created"] * len(wrappers.wrapper_tools())
    for path, _ in results:
        assert path.stat().st_mode & 0o111
        assert "/opt/venv/bin/structbio" in path.read_text(encoding="utf-8")
    assert [state for _, state in wrappers.install_wrappers(
        tmp_path / "bin", launch="/opt/venv/bin/structbio"
    )] == ["unchanged"] * len(wrappers.wrapper_tools())


def test_foreign_files_are_never_replaced_without_force(tmp_path: Path) -> None:
    directory = tmp_path / "bin"
    directory.mkdir()
    (directory / "rfdiffusion").write_text("#!/bin/sh\necho a real installation\n")
    states = {
        path.name: state
        for path, state in wrappers.install_wrappers(directory, launch="structbio")
    }
    assert states["rfdiffusion"].startswith("skipped")
    assert "a real installation" in (directory / "rfdiffusion").read_text()
    states = {
        path.name: state
        for path, state in wrappers.install_wrappers(directory, launch="structbio", force=True)
    }
    assert states["rfdiffusion"] == "updated"
