from pathlib import Path

import pytest
import yaml

from structbio import discovery


def _install_tree(root: Path) -> Path:
    (root / "RFdiffusion" / "scripts").mkdir(parents=True)
    (root / "RFdiffusion" / "scripts" / "run_inference.py").touch()
    (root / "ProteinMPNN").mkdir()
    (root / "ProteinMPNN" / "protein_mpnn_run.py").touch()
    (root / "CryoZeta").mkdir()
    (root / "CryoZeta" / "inference_demo.sh").touch()
    nested = root / "localcolabfold" / ".pixi" / "envs" / "default" / "bin"
    nested.mkdir(parents=True)
    (nested / "colabfold_batch").touch()
    return root


def test_every_backend_can_be_discovered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery, "conda_environments", dict)
    found = discovery.discover(roots=(str(_install_tree(tmp_path)),))
    assert set(found) == {"rfdiffusion", "proteinmpnn", "colabfold", "cryozeta"}
    assert found["rfdiffusion"].path == tmp_path / "RFdiffusion"
    assert found["rfdiffusion"].executable == "scripts/run_inference.py"
    assert found["rfdiffusion"].manager == "conda"
    # ColabFold installs an executable rather than a checkout to point at.
    assert found["colabfold"].path is None
    assert found["colabfold"].executable.endswith("bin/colabfold_batch")


def test_a_directory_without_its_marker_is_not_claimed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(discovery, "conda_environments", dict)
    (tmp_path / "RFdiffusion").mkdir()  # an empty clone, or an unrelated folder
    assert discovery.discover(roots=(str(tmp_path),)) == {}


def test_a_conda_environment_supplies_the_environment_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_tree(tmp_path)
    monkeypatch.setattr(
        discovery, "conda_environments", lambda: {"SE3nv": Path("/envs/SE3nv")}
    )
    found = discovery.discover(roots=(str(tmp_path),))
    assert found["rfdiffusion"].environment == "SE3nv"
    # SE3nv belongs to RFdiffusion; ProteinMPNN is left unset rather than
    # inheriting an environment that would prefix every command wrongly.
    assert found["proteinmpnn"].environment is None


def test_an_executable_on_path_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(discovery, "conda_environments", dict)
    monkeypatch.setattr(discovery.shutil, "which", lambda name: "/usr/bin/colabfold_batch")
    found = discovery.discover(roots=(str(tmp_path),))
    assert found["colabfold"].found_by == "PATH"
    assert found["colabfold"].executable == "colabfold_batch"


def test_the_search_is_bounded(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c" / "d"
    deep.mkdir(parents=True)
    directories = discovery.candidate_directories((str(tmp_path),))
    names = {path.name for path in directories}
    assert {"a", "b"} <= names
    assert "d" not in names  # beyond SEARCH_DEPTH


def test_rendered_config_names_what_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(discovery, "conda_environments", dict)
    (tmp_path / "ProteinMPNN").mkdir()
    (tmp_path / "ProteinMPNN" / "protein_mpnn_run.py").touch()
    text = discovery.render_config(discovery.discover(roots=(str(tmp_path),)))

    payload = yaml.safe_load(text)
    assert set(payload["tools"]) == {"proteinmpnn"}
    assert payload["tools"]["proteinmpnn"]["path"] == str(tmp_path / "ProteinMPNN")
    for absent in ("rfdiffusion", "colabfold", "cryozeta"):
        assert f"# {absent}: not found" in text


def test_merging_never_overwrites_a_configured_tool(tmp_path: Path) -> None:
    existing = "tools:\n  rfdiffusion:\n    path: /my/own/RFdiffusion\n    manager: conda\n"
    found = {
        "rfdiffusion": discovery.Discovery(
            tool="rfdiffusion",
            path=Path("/somewhere/else/RFdiffusion"),
            executable="scripts/run_inference.py",
            manager="conda",
            environment=None,
            found_by="test",
        ),
        "cryozeta": discovery.Discovery(
            tool="cryozeta",
            path=Path("/opt/CryoZeta"),
            executable="inference_demo.sh",
            manager="pixi",
            environment="default",
            found_by="test",
        ),
    }
    merged = yaml.safe_load(discovery.merge_into_config(existing, found))
    assert merged["tools"]["rfdiffusion"]["path"] == "/my/own/RFdiffusion"
    assert merged["tools"]["cryozeta"]["path"] == "/opt/CryoZeta"


def test_merging_nothing_new_leaves_the_file_untouched() -> None:
    existing = "tools:\n  cryozeta:\n    path: /opt/CryoZeta\n"
    found = {
        "cryozeta": discovery.Discovery(
            tool="cryozeta",
            path=Path("/elsewhere"),
            executable="inference_demo.sh",
            manager="pixi",
            environment=None,
            found_by="test",
        )
    }
    assert discovery.merge_into_config(existing, found) == existing


def test_a_config_written_when_nothing_is_found_is_still_usable() -> None:
    """A bare 'tools:' above comments parses as null and breaks every command."""

    from structbio.config import DEFAULT_CONFIG, StructbioSettings, deep_merge

    text = discovery.render_config({})
    payload = yaml.safe_load(text)
    assert payload["tools"] == {}
    StructbioSettings.model_validate(deep_merge(DEFAULT_CONFIG, payload))

    found = {
        "rfdiffusion": discovery.Discovery(
            tool="rfdiffusion",
            path=Path("/opt/RFdiffusion"),
            executable="scripts/run_inference.py",
            manager="conda",
            environment=None,
            found_by="test",
        )
    }
    merged = yaml.safe_load(discovery.merge_into_config(text, found))
    assert merged["tools"]["rfdiffusion"]["path"] == "/opt/RFdiffusion"


def test_a_rendered_config_loads_as_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from structbio.config import load_settings

    monkeypatch.setattr(discovery, "conda_environments", dict)
    text = discovery.render_config(discovery.discover(roots=(str(_install_tree(tmp_path)),)))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(text)
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(config_path))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))

    settings = load_settings()
    assert settings.tools["rfdiffusion"].path == tmp_path / "RFdiffusion"
    assert settings.tools["cryozeta"].manager == "pixi"
