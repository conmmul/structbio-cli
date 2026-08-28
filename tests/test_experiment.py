from datetime import datetime
from pathlib import Path

import pytest

from structbio.experiment import (
    ExperimentManager,
    prepare_output_dir,
    read_metadata,
    write_records,
)


def test_output_directory_collision_never_overwrites(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "experiments")
    first = manager.create("experiment")
    marker = first.root / "raw-marker"
    marker.write_text("preserve me")
    second = manager.create("experiment")
    assert first.root != second.root
    assert marker.read_text() == "preserve me"
    assert first.root.name.endswith("_001")
    assert second.root.name.endswith("_002")


def test_candidate_does_not_create_directory(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "experiments")
    candidate = manager.candidate("dry", datetime(2026, 8, 27))
    assert candidate.name == "dry_2026-08-27_001"
    assert not candidate.exists()


def test_reproducibility_records_include_inputs_outputs_and_command(tmp_path: Path) -> None:
    manager = ExperimentManager(tmp_path / "experiments")
    paths = manager.create("record")
    input_path = tmp_path / "raw.pdb"
    input_path.write_text("RAW")
    metadata = write_records(
        paths,
        config={"tool": "test"},
        command="python tool.py --safe",
        tool_name="test",
        tool_path=None,
        input_paths=[input_path],
        status="prepared",
    )
    assert metadata["command"] == "python tool.py --safe"
    assert metadata["input_paths"] == [str(input_path.resolve())]
    assert metadata["output_paths"] == [str(paths.outputs.resolve())]
    assert metadata["python_version"]
    assert paths.command.read_text() == "python tool.py --safe\n"
    assert input_path.read_text() == "RAW"


def test_output_folder_keeps_results_at_the_top_level(tmp_path: Path) -> None:
    paths = prepare_output_dir(tmp_path / "my_designs")
    assert paths.outputs == tmp_path / "my_designs"
    assert paths.metadata.parent == tmp_path / "my_designs" / ".structbio"
    assert paths.inputs.is_dir()
    assert paths.stdout.is_file()


def test_output_folder_refuses_to_write_over_results(tmp_path: Path) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "design_0.pdb").write_text("ATOM")
    with pytest.raises(ValueError, match="non-empty folder"):
        prepare_output_dir(occupied)
    assert (occupied / "design_0.pdb").read_text() == "ATOM"

    plain_file = tmp_path / "already-a-file"
    plain_file.write_text("data")
    with pytest.raises(ValueError, match="not a folder"):
        prepare_output_dir(plain_file)


def test_output_folder_records_are_readable(tmp_path: Path) -> None:
    paths = prepare_output_dir(tmp_path / "run")
    write_records(
        paths,
        config={"tool": "test"},
        command="python tool.py",
        tool_name="test",
        tool_path=None,
        input_paths=[],
        status="prepared",
    )
    metadata = read_metadata(tmp_path / "run")
    assert metadata is not None
    assert metadata["tool"] == "test"
    assert read_metadata(tmp_path) is None
