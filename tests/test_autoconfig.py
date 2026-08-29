"""Finding a tool the first time it is needed, instead of before it is."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from structbio import autoconfig, discovery


def _checkout(root: Path) -> Path:
    (root / "ProteinMPNN").mkdir(parents=True)
    (root / "ProteinMPNN" / "protein_mpnn_run.py").touch()
    return root


@pytest.fixture
def only_this_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    software = _checkout(tmp_path / "software")
    monkeypatch.setattr(discovery, "conda_environments", dict)
    scan = discovery.discover
    monkeypatch.setattr(discovery, "discover", lambda sigs=None, **_: scan(
        sigs or discovery.SIGNATURES, roots=(str(software),)
    ))
    monkeypatch.delenv(autoconfig.DISABLE_VARIABLE, raising=False)
    return software


def test_find_looks_for_one_tool_only(only_this_machine: Path) -> None:
    assert autoconfig.find("proteinmpnn") is not None
    assert autoconfig.find("rfdiffusion") is None


def test_adopt_writes_a_new_configuration(
    only_this_machine: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config" / "structbio.yaml"
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(config_path))

    adoption = autoconfig.adopt("proteinmpnn")
    assert adoption is not None
    assert adoption.config_path == config_path
    assert adoption.installation.path == only_this_machine / "ProteinMPNN"

    written = yaml.safe_load(config_path.read_text())
    assert written["tools"]["proteinmpnn"]["path"] == str(only_this_machine / "ProteinMPNN")


def test_adopt_never_overwrites_an_entry_you_wrote(
    only_this_machine: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("tools:\n  proteinmpnn:\n    path: /my/ProteinMPNN\n")
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(config_path))

    autoconfig.adopt("proteinmpnn")
    written = yaml.safe_load(config_path.read_text())
    assert written["tools"]["proteinmpnn"]["path"] == "/my/ProteinMPNN"


def test_adopt_still_runs_when_the_configuration_cannot_be_written(
    only_this_machine: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable configuration is bookkeeping lost, not a run refused."""

    unwritable = tmp_path / "config.yaml"
    unwritable.write_text("tools: {}\n")
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(unwritable))
    monkeypatch.setattr(
        Path, "write_text", lambda *a, **k: (_ for _ in ()).throw(OSError("read-only"))
    )

    adoption = autoconfig.adopt("proteinmpnn")
    assert adoption is not None
    assert adoption.config_path is None
    assert adoption.note


def test_it_can_be_switched_off(
    only_this_machine: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(autoconfig.DISABLE_VARIABLE, "1")
    assert not autoconfig.enabled()
    assert autoconfig.adopt("proteinmpnn") is None
