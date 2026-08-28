from pathlib import Path

from structbio import wrappers
from structbio.tools import get_backends


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_repository_wrappers_match_the_generated_template() -> None:
    for tool in wrappers.wrapper_tools():
        path = REPOSITORY_ROOT / "bin" / tool
        assert path.is_file(), f"missing wrapper for {tool}"
        assert path.read_text(encoding="utf-8") == wrappers.render_wrapper(tool)


def test_every_backend_has_a_wrapper() -> None:
    assert wrappers.wrapper_tools() == sorted(get_backends())


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
