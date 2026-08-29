"""Deciding whether a configured tool can actually run."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from structbio import onboard, provision
from structbio.config import StructbioSettings, ToolInstallation


def _working() -> provision.ProbeResult:
    return provision.ProbeResult(
        ok=True,
        values={
            "torch": "2.4.0",
            "torch_cuda": "12.1",
            "cuda_available": True,
            "gpu_allocation": True,
            "device": "NVIDIA RTX 4090",
        },
    )


def _cpu_only() -> provision.ProbeResult:
    return provision.ProbeResult(
        ok=True, values={"torch": "1.9.1.post3", "torch_cuda": None, "cuda_available": False}
    )


def test_a_working_environment_is_reported_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provision, "tool_interpreter", lambda *a: Path("/env/bin/python"))
    monkeypatch.setattr(provision, "verify", lambda *a, **k: _working())

    status = onboard.check("rfdiffusion", ToolInstallation(environment="SE3nv"))
    assert status.ready
    assert status.environment == "SE3nv"
    assert "NVIDIA RTX 4090" in status.detail
    assert status.fix is None


def test_a_pytorch_without_cuda_names_the_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provision, "tool_interpreter", lambda *a: Path("/env/bin/python"))
    monkeypatch.setattr(provision, "verify", lambda *a, **k: _cpu_only())

    status = onboard.check("rfdiffusion", ToolInstallation(environment="SE3nv"))
    assert status.state == onboard.BROKEN
    assert "no usable GPU" in status.detail
    assert status.fix == "structbio env repair rfdiffusion"


def test_a_missing_environment_names_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provision, "tool_interpreter", lambda *a: None)
    monkeypatch.setattr(onboard.environment, "conda_environments", dict)

    status = onboard.check("proteinmpnn", ToolInstallation(environment="mlfold"))
    assert status.state == onboard.NO_ENVIRONMENT
    assert status.fix == "structbio env create proteinmpnn"


def test_an_environment_that_already_works_is_adopted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Somebody's working environment beats a second one built beside it."""

    monkeypatch.setattr(
        onboard.environment, "conda_environments", lambda: {"mlfold": Path("/envs/mlfold")}
    )
    monkeypatch.setattr(provision, "tool_interpreter", lambda *a: Path("/env/bin/python"))
    monkeypatch.setattr(provision, "verify", lambda *a, **k: _working())

    status = onboard.check("proteinmpnn", ToolInstallation())
    assert status.ready
    assert status.adopted
    assert status.environment == "mlfold"


def test_a_tool_that_manages_itself_is_not_probed() -> None:
    status = onboard.check("colabfold", ToolInstallation(executable="colabfold_batch"))
    assert status.state == onboard.UNCHECKED
    assert status.fix is None


def test_recording_an_environment_keeps_a_backup(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("tools:\n  proteinmpnn:\n    path: /my/ProteinMPNN\n")

    written = onboard.record_environment("proteinmpnn", "mlfold", config_path)
    assert written == config_path
    data = yaml.safe_load(config_path.read_text())
    assert data["tools"]["proteinmpnn"]["environment"] == "mlfold"
    assert data["tools"]["proteinmpnn"]["path"] == "/my/ProteinMPNN"
    assert (tmp_path / "config.yaml.bak").exists()

    # Nothing changes, and no second backup is written, when it is already there.
    assert onboard.record_environment("proteinmpnn", "mlfold", config_path) is None


def test_review_only_checks_configured_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provision, "tool_interpreter", lambda *a: Path("/env/bin/python"))
    monkeypatch.setattr(provision, "verify", lambda *a, **k: _working())
    settings = StructbioSettings(tools={"rfdiffusion": ToolInstallation(environment="SE3nv")})

    statuses = onboard.review(settings)
    assert [status.tool for status in statuses] == ["rfdiffusion"]
