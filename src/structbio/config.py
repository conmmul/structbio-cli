"""Layered YAML configuration loading."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_CONFIG: dict[str, Any] = {
    "experiments_root": "./experiments",
    "tools": {
        "rfdiffusion": {"manager": "conda", "executable": "scripts/run_inference.py"},
        "proteinmpnn": {"manager": "conda", "executable": "protein_mpnn_run.py"},
        "cryozeta": {"manager": "pixi", "executable": "inference_demo.sh"},
    },
    "cluster_profiles": {},
}


USER_CONFIG_TEMPLATE = """# structbio workstation configuration.
#
# `path` is the root of an existing tool checkout on this machine and
# `executable` is relative to that root. structbio never installs the
# scientific software or its model weights.
#
# Delete the entries for software this workstation does not have.

tools:
  rfdiffusion:
    path: ~/software/RFdiffusion
    executable: scripts/run_inference.py
    manager: conda
    environment: SE3nv
  proteinmpnn:
    path: ~/software/ProteinMPNN
    executable: protein_mpnn_run.py
    manager: conda
    environment: mlfold
  cryozeta:
    path: ~/software/CryoZeta
    executable: inference_demo.sh
    manager: pixi
    environment: default

# Where `structbio TOOL run CONFIG.yaml` keeps dated experiment folders. Quick
# commands ignore this and write to the output folder you name instead.
experiments_root: ~/structbio-experiments
"""


class ToolInstallation(BaseModel):
    """How a lab installation of an external tool is reached."""

    model_config = ConfigDict(extra="allow")

    path: Path | None = None
    environment: str | None = None
    manager: Literal["conda", "pixi", "none"] = "none"
    executable: str | None = None

    @model_validator(mode="after")
    def expand_path(self) -> "ToolInstallation":
        if self.path is not None:
            self.path = self.path.expanduser()
        return self


class ResourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nodes: int = Field(default=1, ge=1)
    gpus: int = Field(default=0, ge=0)
    cpus: int = Field(default=1, ge=1)
    memory: str = "8G"
    time: str = "01:00:00"
    array: str | None = None


class ClusterProfile(ResourceConfig):
    name: str | None = None
    modules: list[str] = Field(default_factory=list)
    preamble: list[str] = Field(default_factory=list)


class StructbioSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiments_root: Path = Path("./experiments")
    tools: dict[str, ToolInstallation] = Field(default_factory=dict)
    cluster_profiles: dict[str, ClusterProfile] = Field(default_factory=dict)


class LoadedConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: dict[str, Any]
    settings: StructbioSettings
    source: Path
    layers: list[Path]


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; scalars and lists replace earlier values."""

    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root in {path} must be a mapping")
    return raw


def _configured_path(env_name: str, fallback: Path) -> Path:
    value = os.environ.get(env_name)
    return Path(value).expanduser() if value else fallback


def lab_config_path(override: Path | None = None) -> Path:
    return override or _configured_path(
        "STRUCTBIO_LAB_CONFIG", Path("/etc/structbio/config.yaml")
    )


def user_config_path(override: Path | None = None) -> Path:
    return override or _configured_path(
        "STRUCTBIO_USER_CONFIG", Path.home() / ".config/structbio/config.yaml"
    )


def _finalize(data: dict[str, Any], source: Path, layers: list[Path]) -> LoadedConfig:
    settings_data = {
        key: data[key]
        for key in ("experiments_root", "tools", "cluster_profiles")
        if key in data
    }
    settings = StructbioSettings.model_validate(settings_data)
    return LoadedConfig(data=data, settings=settings, source=source, layers=layers)


def load_config(
    experiment_path: Path,
    *,
    lab_config: Path | None = None,
    user_config: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> LoadedConfig:
    """Load defaults < lab < user < experiment < CLI overrides."""

    experiment_path = experiment_path.expanduser().resolve()
    if not experiment_path.is_file():
        raise ValueError(f"Configuration file does not exist: {experiment_path}")

    data = deepcopy(DEFAULT_CONFIG)
    layers: list[Path] = []
    for candidate in (
        lab_config_path(lab_config),
        user_config_path(user_config),
        experiment_path,
    ):
        candidate = candidate.expanduser()
        if candidate.is_file():
            data = deep_merge(data, read_yaml(candidate))
            layers.append(candidate.resolve())
    if overrides:
        data = deep_merge(data, overrides)
    return _finalize(data, experiment_path, layers)


def load_command_line_config(
    fragment: dict[str, Any],
    *,
    lab_config: Path | None = None,
    user_config: Path | None = None,
    overrides: dict[str, Any] | None = None,
    working_dir: Path | None = None,
) -> LoadedConfig:
    """Load defaults < lab < user < arguments typed on the command line.

    Relative paths typed on the command line are resolved against the current
    working directory, so `source` names a file that is never read: only its
    parent directory matters to `resolve_from_config`.
    """

    data = deepcopy(DEFAULT_CONFIG)
    layers: list[Path] = []
    for candidate in (lab_config_path(lab_config), user_config_path(user_config)):
        candidate = candidate.expanduser()
        if candidate.is_file():
            data = deep_merge(data, read_yaml(candidate))
            layers.append(candidate.resolve())
    data = deep_merge(data, fragment)
    if overrides:
        data = deep_merge(data, overrides)
    source = (working_dir or Path.cwd()).resolve() / "command-line"
    return _finalize(data, source, layers)


def load_settings(
    *, lab_config: Path | None = None, user_config: Path | None = None
) -> StructbioSettings:
    """Load defaults plus optional lab and user configuration for diagnostics."""

    data = deepcopy(DEFAULT_CONFIG)
    for candidate in (lab_config_path(lab_config), user_config_path(user_config)):
        candidate = candidate.expanduser()
        if candidate.is_file():
            data = deep_merge(data, read_yaml(candidate))
    return StructbioSettings.model_validate(data)


def resolve_from_config(value: str | Path, source: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    return path.resolve()


def parse_cli_overrides(values: list[str]) -> dict[str, Any]:
    """Parse repeatable `dotted.key=YAML_VALUE` command-line overrides."""

    result: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --set value {item!r}; expected dotted.key=value")
        dotted_key, raw_value = item.split("=", 1)
        keys = dotted_key.split(".")
        if not all(key and key.replace("_", "").replace("-", "").isalnum() for key in keys):
            raise ValueError(f"Invalid --set key: {dotted_key!r}")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML value for --set {dotted_key}: {exc}") from exc
        cursor = result
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            if not isinstance(child, dict):
                raise ValueError(f"Conflicting --set key: {dotted_key!r}")
            cursor = child
        cursor[keys[-1]] = value
    return result
