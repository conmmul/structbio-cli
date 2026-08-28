import json
import re
from pathlib import Path

import yaml

from structbio.tools import get_backend


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_local_markdown_links_resolve() -> None:
    markdown_files = [REPOSITORY_ROOT / "README.md"]
    markdown_files.extend((REPOSITORY_ROOT / "docs").glob("*.md"))
    markdown_files.extend((REPOSITORY_ROOT / "examples").glob("*.md"))
    pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    missing: list[str] = []
    for markdown_file in markdown_files:
        text = markdown_file.read_text(encoding="utf-8")
        for target in pattern.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            path_part = target.split("#", 1)[0]
            if not (markdown_file.parent / path_part).resolve().exists():
                missing.append(f"{markdown_file.relative_to(REPOSITORY_ROOT)} -> {target}")
    assert not missing, "Broken documentation links:\n" + "\n".join(missing)


def test_example_configuration_files_are_parseable() -> None:
    yaml_files = sorted((REPOSITORY_ROOT / "examples").rglob("*.yaml"))
    assert yaml_files
    for path in yaml_files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), path
        if isinstance(payload.get("tool"), str):
            get_backend(payload["tool"]).parse_config(payload, path.resolve())

    json_path = REPOSITORY_ROOT / "examples/cryozeta/native_input.example.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list) and payload


def test_documented_yaml_blocks_are_parseable() -> None:
    markdown_files = [REPOSITORY_ROOT / "README.md"]
    markdown_files.extend((REPOSITORY_ROOT / "docs").glob("*.md"))
    pattern = re.compile(r"```yaml\n(.*?)```", re.DOTALL)
    for markdown_file in markdown_files:
        for index, block in enumerate(
            pattern.findall(markdown_file.read_text(encoding="utf-8")), start=1
        ):
            try:
                payload = yaml.safe_load(block)
            except yaml.YAMLError as exc:
                raise AssertionError(
                    f"Invalid YAML block {index} in {markdown_file.name}: {exc}"
                ) from exc
            assert isinstance(payload, dict), (markdown_file, index)


def test_documented_short_commands_exist() -> None:
    """Every `tool runtype` shown in the README must be a real command."""

    from structbio.cli import app

    groups = {group.name: group.typer_instance for group in app.registered_groups}
    text = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    documented = set(
        re.findall(r"^\| `(rfdiffusion|proteinmpnn|cryozeta) ([a-z-]+)", text, re.MULTILINE)
    )
    assert documented, "no short commands are documented"
    for tool, runtype in sorted(documented):
        commands = {
            command.name or command.callback.__name__
            for command in groups[tool].registered_commands
        }
        assert runtype in commands, f"README documents unknown command: {tool} {runtype}"
