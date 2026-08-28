from pathlib import Path

import pytest

from structbio.config import ToolInstallation
from structbio.tools.base import BackendContext
from structbio.tools.proteinmpnn import (
    ProteinMPNNBackend,
    assert_mask_equivalence,
    inspect_mask,
)
from structbio.validation import StructureValidationError


def _config(backend: ProteinMPNNBackend, tmp_path: Path, tiny_pdb: Path):
    return backend.parse_config(
        {
            "tool": "proteinmpnn",
            "experiment": {"name": "region"},
            "input": {"pdb": str(tiny_pdb)},
            "design": {
                "chains": ["A"],
                "designable_positions": ["A:697-699"],
                "num_sequences": 4,
                "temperatures": [0.1, 0.2],
            },
        },
        tmp_path / "config.yaml",
    )


def test_mask_generation_uses_chain_ordinals(tmp_path: Path, tiny_pdb: Path) -> None:
    backend = ProteinMPNNBackend()
    inspection = inspect_mask(_config(backend, tmp_path, tiny_pdb), tiny_pdb)
    assert {item.number for item in inspection.designable} == {697, 698, 699}
    assert inspection.fixed_dictionary[tiny_pdb.stem] == {
        "A": [4],
        "B": [1, 2, 3],
    }
    assert "A697-A699" in inspection.render()


def test_opposite_fixed_dictionary_aborts(tmp_path: Path, tiny_pdb: Path) -> None:
    backend = ProteinMPNNBackend()
    inspection = inspect_mask(_config(backend, tmp_path, tiny_pdb), tiny_pdb)
    opposite = {"A": [1, 2, 3], "B": []}
    with pytest.raises(StructureValidationError, match="inverts or changes"):
        assert_mask_equivalence(
            inspection.structure,
            inspection.design_chains,
            opposite,
            set(inspection.designable),
        )


def test_missing_chain_validation(tmp_path: Path, tiny_pdb: Path) -> None:
    backend = ProteinMPNNBackend()
    config = backend.parse_config(
        {
            "tool": "proteinmpnn",
            "experiment": {"name": "bad"},
            "input": {"pdb": str(tiny_pdb)},
            "design": {"chains": ["Z"]},
        },
        tmp_path / "config.yaml",
    )
    report = backend.validate(config)
    assert not report.ok
    assert "absent" in report.errors[0]


def test_command_includes_generated_mask(tmp_path: Path, tiny_pdb: Path) -> None:
    backend = ProteinMPNNBackend()
    config = _config(backend, tmp_path, tiny_pdb)
    context = BackendContext(
        source=tmp_path / "config.yaml",
        installation=ToolInstallation(
            path=Path("/opt/ProteinMPNN"),
            environment="SE3nv",
            manager="conda",
            executable="protein_mpnn_run.py",
        ),
        experiment_dir=tmp_path / "experiment",
        output_dir=tmp_path / "experiment/outputs",
        inputs_dir=tmp_path / "experiment/inputs",
    )
    plan = backend.build_command(config, context)
    argv = plan.steps[0].argv
    assert "--fixed_positions_jsonl" in argv
    artifact = next(iter(plan.artifacts["files"].values()))
    assert '"A": [4]' in artifact
