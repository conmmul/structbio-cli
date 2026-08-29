import gzip
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from helpers import write_fasta, write_map
from structbio.config import ToolInstallation
from structbio.tools.base import BackendContext
from structbio.tools.cryozeta import CryoZetaBackend


def test_cryozeta_uses_verified_official_script_interface(tmp_path: Path) -> None:
    write_map(tmp_path / "target.map")
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps(
            [
                {
                    "name": "target",
                    "modelSeeds": [],
                    "map_path": "target.map",
                    "resolution": 3.0,
                    "contour_level": 0.2,
                    "sequences": [{"dnaSequence": {"sequence": "ACGT", "count": 1}}],
                }
            ]
        )
    )
    backend = CryoZetaBackend()
    config = backend.parse_config(
        {
            "tool": "cryozeta",
            "experiment": {"name": "target"},
            "input": {"json": str(input_json)},
            "mode": "combined",
            "pixi_environment": "cu11",
        },
        tmp_path / "dataset.yaml",
    )
    assert backend.validate(config).ok
    context = BackendContext(
        source=tmp_path / "dataset.yaml",
        installation=ToolInstallation(
            path=Path("/opt/CryoZeta"),
            executable="inference_demo.sh",
            manager="pixi",
        ),
        experiment_dir=tmp_path / "experiment",
        output_dir=tmp_path / "experiment/outputs",
        inputs_dir=tmp_path / "experiment/inputs",
    )
    argv = backend.build_command(config, context).steps[0].argv
    assert argv[:2] == ("bash", "/opt/CryoZeta/inference_demo.sh")
    assert "--input-json" in argv
    assert "--output-dir" in argv
    assert "--mode" in argv
    assert "--env" in argv
    assert "--overwrite" not in argv


def _context(tmp_path: Path, path: Path = Path("/opt/CryoZeta")) -> BackendContext:
    return BackendContext(
        source=tmp_path / "dataset.yaml",
        installation=ToolInstallation(
            path=path, executable="inference_demo.sh", manager="pixi"
        ),
        experiment_dir=tmp_path / "out",
        output_dir=tmp_path / "out",
        inputs_dir=tmp_path / "out" / ".structbio" / "inputs",
    )


def _built(tmp_path: Path, backend: CryoZetaBackend, **overrides: object):
    # A test may have written its own map or FASTA first; do not clobber it.
    density_map = tmp_path / "target.map"
    if not density_map.exists():
        write_map(density_map)
    chains = tmp_path / "chains.fasta"
    if not chains.exists():
        write_fasta(chains)
    source = {
        "map": str(density_map),
        "sequences": str(chains),
        "resolution": 2.99,
        "contour_level": 0.3,
    }
    source.update(overrides.pop("input", {}))  # type: ignore[arg-type]
    raw = {
        "tool": "cryozeta",
        "experiment": {"name": "my_model"},
        "input": source,
    }
    raw.update(overrides)
    return backend.parse_config(raw, tmp_path / "dataset.yaml")


def test_native_json_is_generated_from_a_map_and_a_fasta(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    write_fasta(
        tmp_path / "chains.fasta",
        ">chain_A\nMKTAYIAKQ\n>chain_B\nMKTAYIAKQ\n>rna_1\nACGUACGU\n",
    )
    config = _built(tmp_path, backend)
    report = backend.validate(config)
    assert report.ok, report.errors

    context = _context(tmp_path)
    plan = backend.build_command(config, context)
    generated = context.inputs_dir / "targets.json"
    payload = json.loads(plan.artifacts["files"][str(generated)])

    assert payload == [
        {
            "name": "my_model",
            "modelSeeds": [],
            "map_path": str(tmp_path / "target.map"),
            "resolution": 2.99,
            "contour_level": 0.3,
            "sequences": [
                # Identical chains become one entry with a count.
                {"proteinChain": {"sequence": "MKTAYIAKQ", "count": 2}},
                {"rnaSequence": {"sequence": "ACGUACGU", "count": 1}},
            ],
        }
    ]
    assert plan.steps[0].argv[3] == str(generated)


def test_an_ambiguous_sequence_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """A, C, G and T are valid amino acids as well as nucleotides."""

    backend = CryoZetaBackend()
    write_fasta(tmp_path / "chains.fasta", ">chain_A\nACGTACGTACGT\n")
    report = backend.validate(_built(tmp_path, backend))
    assert not report.ok
    assert any("Cannot tell whether" in error for error in report.errors)


def test_a_declared_chain_type_resolves_the_ambiguity(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    write_fasta(tmp_path / "chains.fasta", ">chain_A\nACGTACGTACGT\n")
    config = _built(tmp_path, backend, input={"chains": {"dna": ["chain_A"]}})
    assert backend.validate(config).ok
    assert "dnaSequence" in backend.targets(config)[0]["sequences"][0]


def test_msa_settings_are_attached_to_protein_and_rna_only(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    write_fasta(
        tmp_path / "chains.fasta", ">protein\nMKTAYIAKQ\n>dna\nACGTACGT\n"
    )
    msa_dir = tmp_path / "msas"
    msa_dir.mkdir()
    config = _built(
        tmp_path,
        backend,
        input={
            "chains": {"dna": ["dna"]},
            "msa": {"precomputed_msa_dir": str(msa_dir), "pairing_db": "uniref100"},
        },
    )
    sequences = backend.targets(config)[0]["sequences"]
    assert sequences[0]["proteinChain"]["msa"] == {
        "precomputed_msa_dir": str(msa_dir),
        "pairing_db": "uniref100",
    }
    assert "msa" not in sequences[1]["dnaSequence"]


def test_a_file_that_is_not_a_density_map_is_rejected(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    (tmp_path / "target.map").write_bytes(b"not a map" * 200)
    report = backend.validate(_built(tmp_path, backend))
    assert not report.ok
    assert any("not an MRC/CCP4" in error for error in report.errors)


def test_a_truncated_density_map_is_rejected(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    (tmp_path / "target.map").write_bytes(b"MAP ")
    report = backend.validate(_built(tmp_path, backend))
    assert not report.ok
    assert any("truncated" in error for error in report.errors)


def test_a_gzipped_map_is_read(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    plain = write_map(tmp_path / "plain.map")
    gzipped = tmp_path / "target.map.gz"
    with gzip.open(gzipped, "wb") as handle:
        handle.write(plain.read_bytes())
    config = _built(tmp_path, backend, input={"map": str(gzipped)})
    report = backend.validate(config)
    assert report.ok, report.errors
    assert any("gzip compressed" in detail for detail in report.details)


def test_a_large_complex_is_steered_to_the_large_pipeline(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    write_fasta(tmp_path / "chains.fasta", ">chain_A\n" + "MKTAYIAKQ" * 400 + "\n")
    report = backend.validate(_built(tmp_path, backend))
    assert report.ok
    assert any("--large" in warning for warning in report.warnings)

    large = backend.validate(_built(tmp_path, backend, large=True))
    assert not any("--large" in warning for warning in large.warnings)


def test_the_large_pipeline_uses_its_own_script_and_flags(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    config = _built(tmp_path, backend, large=True, registration="vesper")
    argv = backend.build_command(config, _context(tmp_path)).steps[0].argv
    assert argv[:2] == ("bash", "/opt/CryoZeta/large_inference_demo.sh")
    assert "--registration" in argv
    assert argv[argv.index("--registration") + 1] == "vesper"
    # large_inference_demo.sh has no --mode and no --interp-checkpoint.
    assert "--mode" not in argv
    assert "--interp-checkpoint" not in argv


def test_checkpoints_that_belong_to_the_other_pipeline_are_rejected(
    tmp_path: Path,
) -> None:
    backend = CryoZetaBackend()
    checkpoint = tmp_path / "weights.safetensors"
    checkpoint.write_bytes(b"weights")
    with pytest.raises(ValidationError, match="does not apply to the large-complex"):
        _built(tmp_path, backend, large=True, interpolation_checkpoint=str(checkpoint))
    with pytest.raises(ValidationError, match="only to the large-complex"):
        _built(tmp_path, backend, detection_checkpoint=str(checkpoint))


def test_a_missing_checkpoint_is_reported(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    config = _built(tmp_path, backend, checkpoint=str(tmp_path / "absent.safetensors"))
    report = backend.validate(config)
    assert not report.ok
    assert any("checkpoint does not exist" in error for error in report.errors)


def test_hand_written_sequence_entries_are_checked(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    write_map(tmp_path / "target.map")
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps(
            [
                {
                    "name": "target",
                    "modelSeeds": [],
                    "map_path": "target.map",
                    "resolution": 3.0,
                    "contour_level": 0.2,
                    "sequences": [
                        {"proteinChain": {"sequence": "", "count": 1}},
                        {"peptide": {"sequence": "MKT", "count": 1}},
                        {"dnaSequence": {"sequence": "ACGT"}},
                    ],
                }
            ]
        )
    )
    config = backend.parse_config(
        {
            "tool": "cryozeta",
            "experiment": {"name": "target"},
            "input": {"json": str(input_json)},
        },
        tmp_path / "dataset.yaml",
    )
    report = backend.validate(config)
    assert not report.ok
    joined = " ".join(report.errors)
    assert "non-empty 'sequence' string" in joined
    assert "unknown type 'peptide'" in joined
    assert "'count' of at least 1" in joined


def test_exactly_one_input_source_is_required(tmp_path: Path) -> None:
    backend = CryoZetaBackend()
    with pytest.raises(ValidationError, match=r"Set either input\.json"):
        backend.parse_config(
            {"tool": "cryozeta", "experiment": {"name": "x"}, "input": {}},
            tmp_path / "dataset.yaml",
        )
    with pytest.raises(ValidationError, match="needs input"):
        backend.parse_config(
            {
                "tool": "cryozeta",
                "experiment": {"name": "x"},
                "input": {"map": str(write_map(tmp_path / "target.map"))},
            },
            tmp_path / "dataset.yaml",
        )


def test_target_chains_counts_every_copy(tmp_path: Path) -> None:
    from structbio.tools.cryozeta import target_chains

    backend = CryoZetaBackend()
    write_fasta(
        tmp_path / "chains.fasta",
        ">a\nMKTAYIAKQ\n>b\nMKTAYIAKQ\n>c\nGGSGGSGGS\n",
    )
    # Two identical chains collapse to one entry with count 2, plus one other:
    # three chains in the model, matching the three records given.
    assert target_chains(_built(tmp_path, backend)) == 3
