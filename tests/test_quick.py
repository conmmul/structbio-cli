from pathlib import Path

import pytest

from structbio import quick
from structbio.validation import StructureValidationError, parse_pdb


def test_symmetry_order_covers_supported_groups() -> None:
    assert quick.symmetry_order("c4") == 4
    assert quick.symmetry_order("D3") == 6
    assert quick.symmetry_order("tetrahedral") == 12
    with pytest.raises(ValueError, match="Unsupported symmetry"):
        quick.symmetry_order("icosahedral")


def test_symmetry_length_must_divide_by_subunit_count() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        quick.rfdiffusion_symmetry(name="x", symmetry="c4", length=150, num_designs=1)
    fragment = quick.rfdiffusion_symmetry(name="x", symmetry="c4", length=400, num_designs=1)
    assert fragment["design"]["length"] == 400


def test_binder_contig_uses_pdb_numbering(tiny_pdb: Path) -> None:
    assert quick.binder_contig(tiny_pdb, "A", 100) == "A697-700/0 100-100"
    assert quick.binder_contig(tiny_pdb, "B", 80) == "B10-12/0 80-80"


def test_binder_requires_a_named_chain_for_multi_chain_input(tiny_pdb: Path) -> None:
    with pytest.raises(StructureValidationError, match="name the target chain"):
        quick.binder_contig(tiny_pdb, None, 100)
    with pytest.raises(StructureValidationError, match="absent"):
        quick.binder_contig(tiny_pdb, "Z", 100)


def test_contig_segments_split_at_numbering_gaps(tmp_path: Path, tiny_pdb: Path) -> None:
    gapped = tmp_path / "gapped.pdb"
    lines = [
        f"ATOM  {index:5d}  CA  ALA A{number:4d}      "
        "11.104  13.207   9.180  1.00 20.00           C"
        for index, number in enumerate([1, 2, 3, 10, 11], start=1)
    ]
    gapped.write_text("\n".join(lines) + "\n")
    assert quick.binder_contig(gapped, "A", 50) == "A1-3/A10-11/0 50-50"
    assert quick.contiguous_segments(parse_pdb(tiny_pdb).for_chain("A")) == [(697, 700)]


def test_partial_contig_keeps_every_chain_at_its_own_length(tiny_pdb: Path) -> None:
    fragment = quick.rfdiffusion_partial(name="x", pdb=tiny_pdb, steps=10, num_designs=3)
    assert fragment["design"]["contigs"] == ["4-4/0 3-3"]
    assert fragment["diffusion"]["partial_t"] == 10


def test_hotspots_are_checked_before_the_tool_runs(tiny_pdb: Path) -> None:
    with pytest.raises(ValueError, match="Invalid hotspot"):
        quick.rfdiffusion_binder(
            name="x", target=tiny_pdb, length=50, num_designs=1, chain="A", hotspots="B30,oops"
        )


def test_proteinmpnn_detects_directory_input(tmp_path: Path, tiny_pdb: Path) -> None:
    folder = tmp_path / "structures"
    folder.mkdir()
    assert "directory" in quick.proteinmpnn_design(
        name="x", input_path=folder, num_sequences=4
    )["input"]
    assert "pdb" in quick.proteinmpnn_design(
        name="x", input_path=tiny_pdb, num_sequences=4
    )["input"]


def test_option_lists_and_gpu_ids_are_parsed() -> None:
    assert quick.split_list("A,B C") == ["A", "B", "C"]
    assert quick.split_list(None) == []
    assert quick.parse_gpu_ids("0,1") == [0, 1]
    with pytest.raises(ValueError, match="Invalid GPU selection"):
        quick.parse_gpu_ids("cuda:0")
