from pathlib import Path

import pytest

from structbio.config import deep_merge, load_config, parse_cli_overrides, read_yaml


def test_deep_merge_replaces_lists_and_merges_mappings() -> None:
    result = deep_merge(
        {"tools": {"x": {"path": "/a", "environment": "old"}}, "items": [1]},
        {"tools": {"x": {"environment": "new"}}, "items": [2]},
    )
    assert result == {
        "tools": {"x": {"path": "/a", "environment": "new"}},
        "items": [2],
    }


def test_configuration_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lab = tmp_path / "lab.yaml"
    user = tmp_path / "user.yaml"
    experiment = tmp_path / "experiment.yaml"
    lab.write_text("tools:\n  rfdiffusion:\n    environment: lab\n    path: /lab/rf\n")
    user.write_text("tools:\n  rfdiffusion:\n    environment: user\n")
    experiment.write_text(
        "tool: rfdiffusion\nexperiment: {name: test}\ndesign: {length: 100}\n"
        "tools:\n  rfdiffusion:\n    environment: experiment\n"
    )
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(lab))
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user))
    loaded = load_config(
        experiment, overrides={"tools": {"rfdiffusion": {"environment": "cli"}}}
    )
    installation = loaded.settings.tools["rfdiffusion"]
    assert installation.environment == "cli"
    assert installation.path == Path("/lab/rf")


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        read_yaml(path)


def test_parse_cli_dotted_overrides() -> None:
    assert parse_cli_overrides(
        ["design.num_designs=40", "resources.gpus=4", "design.soluble_model=true"]
    ) == {
        "design": {"num_designs": 40, "soluble_model": True},
        "resources": {"gpus": 4},
    }
