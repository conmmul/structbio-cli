from pathlib import Path

import pytest

from structbio.config import ToolInstallation
from structbio.environment import diagnose_installation, executable_path, git_commit
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


def _installation(**kwargs: object) -> ToolInstallation:
    return ToolInstallation(**kwargs)  # type: ignore[arg-type]


def test_a_missing_path_is_named_with_a_fix(tmp_path: Path) -> None:
    problems, remedies = diagnose_installation(
        _installation(path=tmp_path / "absent", executable="scripts/run_inference.py"),
        tool="rfdiffusion",
        default_executable="scripts/run_inference.py",
    )
    assert any("does not exist" in problem for problem in problems)
    assert any("structbio install rfdiffusion" in remedy for remedy in remedies)


def test_an_incomplete_checkout_is_distinguished_from_a_missing_one(tmp_path: Path) -> None:
    (tmp_path / "RFdiffusion").mkdir()
    problems, _ = diagnose_installation(
        _installation(path=tmp_path / "RFdiffusion", executable="scripts/run_inference.py"),
        tool="rfdiffusion",
        default_executable="scripts/run_inference.py",
    )
    assert any("does not contain scripts/run_inference.py" in problem for problem in problems)
    assert not any("does not exist" in problem for problem in problems)


def test_an_executable_expected_on_path_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("structbio.environment.shutil.which", lambda name: None)
    problems, remedies = diagnose_installation(
        _installation(executable="colabfold_batch", manager="none"),
        tool="colabfold",
        default_executable="colabfold_batch",
    )
    assert any("not on PATH" in problem for problem in problems)
    assert any("structbio install colabfold" in remedy for remedy in remedies)


def test_a_missing_conda_environment_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "scripts" / "run_inference.py"
    script.parent.mkdir()
    script.touch()
    monkeypatch.setattr("structbio.environment.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "structbio.environment.conda_environments", lambda: {"other": Path("/envs/other")}
    )
    installation = _installation(
        path=tmp_path,
        executable="scripts/run_inference.py",
        manager="conda",
        environment="SE3nv",
    )
    problems, remedies = diagnose_installation(
        installation, tool="rfdiffusion", default_executable="scripts/run_inference.py"
    )
    assert any("'SE3nv' does not exist" in problem for problem in problems)
    assert remedies

    # The tool is present but unusable, so it must not be reported as found.
    check = RFDiffusionBackend().check_environment(installation)
    assert check.configured
    assert not check.found
    assert "SE3nv" in check.explain()


def test_a_complete_installation_reports_no_problems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "scripts" / "run_inference.py"
    script.parent.mkdir()
    script.touch()
    monkeypatch.setattr(
        "structbio.environment.conda_environments", lambda: {"SE3nv": Path("/envs/SE3nv")}
    )
    monkeypatch.setattr("structbio.environment.shutil.which", lambda name: f"/usr/bin/{name}")
    check = RFDiffusionBackend().check_environment(
        _installation(
            path=tmp_path,
            executable="scripts/run_inference.py",
            manager="conda",
            environment="SE3nv",
        )
    )
    assert check.found
    assert check.problems == ()


def test_missing_pixi_is_reported_for_cryozeta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from structbio.tools.cryozeta import CryoZetaBackend

    (tmp_path / "inference_demo.sh").touch()
    monkeypatch.setattr("structbio.environment.shutil.which", lambda name: None)
    check = CryoZetaBackend().check_environment(
        _installation(path=tmp_path, executable="inference_demo.sh", manager="pixi")
    )
    assert not check.found
    assert any("pixi is not installed" in problem for problem in check.problems)
    assert any("pixi.sh/install.sh" in remedy for remedy in check.remedies)
