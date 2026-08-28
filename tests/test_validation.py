from pathlib import Path

import pytest

from structbio.validation import (
    StructureValidationError,
    parse_fasta,
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


def test_parse_fasta_splits_complex_chains(tmp_path: Path) -> None:
    path = tmp_path / "designs.fa"
    path.write_text(">design_0 score=1.2\nMKTAYIAK\nQRQI\n\n>design_1\nMKTA:GGGS\n")
    records = parse_fasta(path)
    assert [record.name for record in records] == ["design_0 score=1.2", "design_1"]
    assert records[0].chains == ("MKTAYIAKQRQI",)
    assert records[1].chains == ("MKTA", "GGGS")
    assert records[1].length == 8


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (">design\nMKTAYIAK1\n", "non-amino-acid"),
        (">design\n\n", "no sequence"),
        (">design\nMKTA::GGGS\n", "empty chain"),
        ("MKTAYIAK\n", "before the first"),
        (">\nMKTAYIAK\n", "Unnamed FASTA record"),
        ("\n", "No FASTA records"),
    ],
)
def test_malformed_fasta_is_rejected(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "bad.fa"
    path.write_text(text)
    with pytest.raises(StructureValidationError, match=message):
        parse_fasta(path)


def test_map_header_is_read_with_either_byte_order(tmp_path: Path) -> None:
    import struct

    from structbio.validation import parse_map_header

    for order, stamp in (("<", b"\x44\x44"), (">", b"\x11\x11")):
        header = bytearray(1024)
        header[0:16] = struct.pack(f"{order}4i", 90, 100, 110, 2)
        header[208:212] = b"MAP "
        header[212:214] = stamp
        path = tmp_path / f"map_{order == '<'}.map"
        path.write_bytes(bytes(header))
        parsed = parse_map_header(path)
        assert (parsed.columns, parsed.rows, parsed.sections) == (90, 100, 110)
        assert parsed.voxels == 90 * 100 * 110


def test_map_header_rejects_impossible_grids(tmp_path: Path) -> None:
    import struct

    from structbio.validation import parse_map_header

    header = bytearray(1024)
    header[0:16] = struct.pack("<4i", 0, 100, 110, 2)
    header[208:212] = b"MAP "
    header[212:214] = b"\x44\x44"
    path = tmp_path / "bad.map"
    path.write_bytes(bytes(header))
    with pytest.raises(StructureValidationError, match="impossible grid"):
        parse_map_header(path)


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("MKTAYIAKQ", "protein"),
        ("ACGUACGU", "rna"),
        ("ACGTACGT", "ambiguous"),
        ("ACGACG", "ambiguous"),
        ("", "ambiguous"),
    ],
)
def test_sequence_classification_refuses_to_guess(sequence: str, expected: str) -> None:
    from structbio.validation import classify_sequence

    assert classify_sequence(sequence) == expected
