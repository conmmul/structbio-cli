from pathlib import Path

from structbio.config import ToolInstallation
from structbio.environment import executable_path, git_commit
from structbio.tools.rfdiffusion import RFDiffusionBackend


def test_environment_detection_from_configured_path(tmp_path: Path) -> None:
    script = tmp_path / "scripts/run_inference.py"
    script.parent.mkdir()
    script.write_text("#!/usr/bin/env python\n")
    installation = ToolInstallation(
        path=tmp_path,
        manager="none",
        executable="scripts/run_inference.py",
    )
    assert executable_path(installation) == script.resolve()
    check = RFDiffusionBackend().check_environment(installation)
    assert check.configured
    assert check.found


def test_git_commit_is_none_outside_a_checkout(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None
    assert git_commit(Path(__file__).resolve().parents[1]) is not None
