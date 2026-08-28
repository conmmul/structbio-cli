from pathlib import Path

import pytest

from structbio.validation import (
    StructureValidationError,
    parse_pdb,
    parse_position_spec,
    validate_contig,
)


def test_parse_pdb_preserves_chains_and_numbering(tiny_pdb: Path) -> None:
    structure = parse_pdb(tiny_pdb)
    assert structure.chains == {"A", "B"}
    assert [item.number for item in structure.for_chain("A")] == [697, 698, 699, 700]


def test_invalid_residue_range_is_rejected(tiny_pdb: Path) -> None:
    structure = parse_pdb(tiny_pdb)
    with pytest.raises(StructureValidationError, match="absent residue"):
        parse_position_spec("A:697-701", structure)
    with pytest.raises(StructureValidationError, match="reversed"):
        parse_position_spec("A:700-697", structure)


def test_missing_chain_is_rejected(tiny_pdb: Path) -> None:
    with pytest.raises(StructureValidationError, match="absent"):
        parse_position_spec("Z:1", parse_pdb(tiny_pdb))


def test_malformed_pdb_is_rejected(fixture_dir: Path) -> None:
    with pytest.raises(StructureValidationError, match="no ATOM/HETATM"):
        parse_pdb(fixture_dir / "malformed.pdb")


@pytest.mark.parametrize(
    "contig", ["150-150", "5-15/A697-699/30-40", "B10-12/0 100-100"]
)
def test_valid_contigs(contig: str) -> None:
    validate_contig(contig)


def test_malformed_contig_is_rejected() -> None:
    with pytest.raises(StructureValidationError):
        validate_contig("A10--20")
