from pathlib import Path

import pytest
from pydantic import ValidationError

from structbio.config import ToolInstallation
from structbio.tools.base import BackendContext
from structbio.tools.colabfold import ColabFoldBackend, resolve_sequence_source


def _fasta(path: Path, text: str = ">design_0\nMKTAYIAKQRQISFVKSHFSRQ\n") -> Path:
    path.write_text(text)
    return path


def _config(backend: ColabFoldBackend, tmp_path: Path, **overrides: object):
    sequences = tmp_path / "designs.fa"
    if not sequences.exists():  # a test may have written its own first
        _fasta(sequences)
    raw = {
        "tool": "colabfold",
        "experiment": {"name": "folds"},
        "input": {"sequences": str(sequences)},
    }
    raw.update(overrides)
    return backend.parse_config(raw, tmp_path / "config.yaml")


def _context(tmp_path: Path) -> BackendContext:
    return BackendContext(
        source=tmp_path / "config.yaml",
        installation=ToolInstallation(manager="none", executable="colabfold_batch"),
        experiment_dir=tmp_path / "out",
        output_dir=tmp_path / "out",
        inputs_dir=tmp_path / "out" / "inputs",
    )


def test_command_matches_the_verified_colabfold_interface(tmp_path: Path) -> None:
    backend = ColabFoldBackend()
    config = _config(
        backend,
        tmp_path,
        prediction={"num_models": 3, "num_recycle": 6, "model_type": "alphafold2_ptm"},
        msa={"mode": "single_sequence", "templates": True},
        relax={"num_relax": 1, "use_gpu": True},
        output={"rank": "plddt"},
    )
    plan = backend.build_command(config, _context(tmp_path))
    argv = plan.steps[0].argv

    # colabfold_batch takes the input and the results directory positionally.
    assert argv[0] == "colabfold_batch"
    assert argv[1] == str(tmp_path / "designs.fa")
    assert argv[2] == str(tmp_path / "out")
    rendered = " ".join(argv[3:])
    assert "--num-models 3" in rendered
    assert "--num-recycle 6" in rendered
    assert "--model-type alphafold2_ptm" in rendered
    assert "--msa-mode single_sequence" in rendered
    assert "--templates" in rendered
    assert "--num-relax 1" in rendered
    assert "--use-gpu-relax" in rendered
    assert "--rank plddt" in rendered


def test_defaults_stay_off_the_command_line(tmp_path: Path) -> None:
    backend = ColabFoldBackend()
    plan = backend.build_command(_config(backend, tmp_path), _context(tmp_path))
    rendered = " ".join(plan.steps[0].argv)
    assert "--num-models 5" in rendered
    for absent in ("--templates", "--num-relax", "--rank", "--use-dropout", "--zip"):
        assert absent not in rendered


def test_the_overwrite_flag_is_never_generated(tmp_path: Path) -> None:
    backend = ColabFoldBackend()
    plan = backend.build_command(_config(backend, tmp_path), _context(tmp_path))
    assert "--overwrite-existing-results" not in " ".join(plan.steps[0].argv)


def test_relaxing_more_models_than_are_predicted_is_rejected(tmp_path: Path) -> None:
    backend = ColabFoldBackend()
    with pytest.raises(ValidationError, match="cannot exceed"):
        _config(backend, tmp_path, prediction={"num_models": 2}, relax={"num_relax": 3})
    with pytest.raises(ValidationError, match="no effect"):
        _config(backend, tmp_path, relax={"num_relax": 0, "use_gpu": True})


def test_public_msa_server_is_flagged_because_sequences_leave_the_machine(
    tmp_path: Path,
) -> None:
    backend = ColabFoldBackend()
    report = backend.validate(_config(backend, tmp_path))
    assert report.ok
    assert any("leaves this machine" in warning for warning in report.warnings)

    local = backend.validate(_config(backend, tmp_path, msa={"mode": "single_sequence"}))
    assert not any("leaves this machine" in warning for warning in local.warnings)


def test_bad_sequences_are_caught_before_a_fold_starts(tmp_path: Path) -> None:
    backend = ColabFoldBackend()
    _fasta(tmp_path / "designs.fa", ">design_0\nMKTAYIAK123\n")
    report = backend.validate(_config(backend, tmp_path))
    assert not report.ok
    assert any("non-amino-acid" in error for error in report.errors)


def test_duplicate_job_names_are_reported(tmp_path: Path) -> None:
    backend = ColabFoldBackend()
    _fasta(tmp_path / "designs.fa", ">same\nMKTAYIAK\n>same\nMKTAYIAKQ\n")
    report = backend.validate(_config(backend, tmp_path))
    assert any("overwrite each other" in warning for warning in report.warnings)


def test_complexes_are_counted_from_the_colon_separator(tmp_path: Path) -> None:
    backend = ColabFoldBackend()
    _fasta(tmp_path / "designs.fa", ">complex\nMKTAYIAK:GGGSGGGS\n")
    report = backend.validate(_config(backend, tmp_path))
    assert "Complexes: 1" in report.details


def test_a_proteinmpnn_output_folder_is_accepted_directly(tmp_path: Path) -> None:
    """RFdiffusion -> ProteinMPNN -> ColabFold should need no manual path fixing."""

    mpnn_output = tmp_path / "my_sequences"
    (mpnn_output / "seqs").mkdir(parents=True)
    _fasta(mpnn_output / "seqs" / "design.fa")
    assert resolve_sequence_source(mpnn_output) == mpnn_output / "seqs"

    backend = ColabFoldBackend()
    config = backend.parse_config(
        {
            "tool": "colabfold",
            "experiment": {"name": "folds"},
            "input": {"sequences": str(mpnn_output)},
        },
        tmp_path / "config.yaml",
    )
    report = backend.validate(config)
    assert report.ok
    assert any("Using designed sequences from" in detail for detail in report.details)
    assert backend.build_command(config, _context(tmp_path)).steps[0].argv[1] == str(
        mpnn_output / "seqs"
    )


def test_a_plain_fasta_folder_is_used_as_given(tmp_path: Path) -> None:
    folder = tmp_path / "sequences"
    folder.mkdir()
    _fasta(folder / "a.fa")
    (folder / "seqs").mkdir()
    assert resolve_sequence_source(folder) == folder


def test_missing_input_is_reported(tmp_path: Path) -> None:
    backend = ColabFoldBackend()
    config = backend.parse_config(
        {
            "tool": "colabfold",
            "experiment": {"name": "folds"},
            "input": {"sequences": str(tmp_path / "absent.fa")},
        },
        tmp_path / "config.yaml",
    )
    report = backend.validate(config)
    assert not report.ok
    assert any("does not exist" in error for error in report.errors)


def test_a_proteinmpnn_directory_batch_is_merged_into_one_input(tmp_path: Path) -> None:
    """A batch writes OUTPUT/STRUCTURE/seqs/*.fa, one level deeper than a single run."""

    mpnn_output = tmp_path / "my_sequences"
    for index, residue in enumerate("AC"):
        seqs = mpnn_output / f"backbone_{index}" / "seqs"
        seqs.mkdir(parents=True)
        (seqs / f"backbone_{index}.fa").write_text(
            f">backbone_{index}\nMKTAYIAK\n>backbone_{index}_sample_0\nMKTAYIA{residue}\n"
        )

    backend = ColabFoldBackend()
    config = backend.parse_config(
        {
            "tool": "colabfold",
            "experiment": {"name": "folds"},
            "input": {"sequences": str(mpnn_output)},
        },
        tmp_path / "config.yaml",
    )
    report = backend.validate(config)
    assert report.ok, report.errors
    assert any("2 ProteinMPNN result folders" in detail for detail in report.details)
    assert "Sequences: 4" in report.details

    context = _context(tmp_path)
    plan = backend.build_command(config, context)
    merged = context.inputs_dir / "sequences.fa"
    assert plan.steps[0].argv[1] == str(merged)

    # Records are copied verbatim: nothing is renamed, reordered, or rewrapped.
    text = plan.artifacts["files"][str(merged)]
    assert text.count(">") == 4
    assert ">backbone_0_sample_0\nMKTAYIAA\n" in text
    assert ">backbone_1_sample_0\nMKTAYIAC\n" in text

    context.inputs_dir.mkdir(parents=True)
    backend.materialize_artifacts(plan)
    assert merged.read_text() == text
    with pytest.raises(FileExistsError):
        backend.materialize_artifacts(plan)


def test_a_single_structure_proteinmpnn_run_needs_no_merging(tmp_path: Path) -> None:
    mpnn_output = tmp_path / "my_sequences"
    (mpnn_output / "seqs").mkdir(parents=True)
    _fasta(mpnn_output / "seqs" / "design.fa")

    backend = ColabFoldBackend()
    config = backend.parse_config(
        {
            "tool": "colabfold",
            "experiment": {"name": "folds"},
            "input": {"sequences": str(mpnn_output)},
        },
        tmp_path / "config.yaml",
    )
    plan = backend.build_command(config, _context(tmp_path))
    assert plan.steps[0].argv[1] == str(mpnn_output / "seqs")
    assert not plan.artifacts["files"]
