"""Structure-file parsing and residue-selection validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class StructureValidationError(ValueError):
    pass


@dataclass(frozen=True, order=True)
class ResidueId:
    chain: str
    number: int
    insertion_code: str = ""

    def label(self) -> str:
        return f"{self.chain}{self.number}{self.insertion_code}"


@dataclass(frozen=True)
class StructureIndex:
    path: Path
    residues: tuple[ResidueId, ...]

    @property
    def chains(self) -> set[str]:
        return {residue.chain for residue in self.residues}

    def for_chain(self, chain: str) -> tuple[ResidueId, ...]:
        return tuple(residue for residue in self.residues if residue.chain == chain)


def parse_pdb(path: Path) -> StructureIndex:
    """Parse residue identifiers without changing chain IDs or numbering."""

    if not path.is_file():
        raise StructureValidationError(f"PDB file does not exist: {path}")
    seen: set[ResidueId] = set()
    residues: list[ResidueId] = []
    atom_lines = 0
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise StructureValidationError(f"Cannot read PDB file {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_lines += 1
        if len(line) < 27:
            raise StructureValidationError(
                f"Malformed PDB atom record at {path}:{line_number}: line is too short"
            )
        chain = line[21]
        if not chain.strip():
            chain = " "
        number_text = line[22:26].strip()
        try:
            number = int(number_text)
        except ValueError as exc:
            raise StructureValidationError(
                f"Malformed residue number at {path}:{line_number}: {number_text!r}"
            ) from exc
        residue = ResidueId(chain=chain, number=number, insertion_code=line[26].strip())
        if residue not in seen:
            seen.add(residue)
            residues.append(residue)
    if atom_lines == 0 or not residues:
        raise StructureValidationError(f"Malformed PDB file {path}: no ATOM/HETATM records")
    return StructureIndex(path=path, residues=tuple(residues))


_POSITION_RE = re.compile(
    r"^(?P<chain>[^:\s]):(?P<start>-?\d+)(?P<start_i>[A-Za-z]?)"
    r"(?:-(?P<end>-?\d+)(?P<end_i>[A-Za-z]?))?$"
)


def parse_position_spec(spec: str, structure: StructureIndex) -> set[ResidueId]:
    """Resolve `A:10-20` or `A:10` against residues actually in a PDB."""

    match = _POSITION_RE.fullmatch(spec.strip())
    if not match:
        raise StructureValidationError(
            f"Invalid residue selection {spec!r}; expected CHAIN:START[-END]"
        )
    chain = match.group("chain")
    if chain not in structure.chains:
        raise StructureValidationError(
            f"Chain {chain!r} requested by {spec!r} is absent from {structure.path.name}"
        )
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    if end < start:
        raise StructureValidationError(f"Residue range is reversed: {spec!r}")
    start_i = match.group("start_i")
    end_i = match.group("end_i")
    if (start_i or end_i) and start != end:
        raise StructureValidationError(
            f"Insertion codes are only supported for single positions: {spec!r}"
        )
    selected = {
        residue
        for residue in structure.for_chain(chain)
        if start <= residue.number <= end
        and (not start_i or residue.insertion_code == start_i)
        and (not end_i or residue.insertion_code == end_i)
    }
    if not selected:
        raise StructureValidationError(
            f"Selection {spec!r} contains no residues in {structure.path.name}"
        )
    present_numbers = {residue.number for residue in selected}
    expected = set(range(start, end + 1))
    missing = sorted(expected - present_numbers)
    if missing:
        preview = ", ".join(map(str, missing[:8]))
        suffix = "…" if len(missing) > 8 else ""
        raise StructureValidationError(
            f"Selection {spec!r} includes absent residue numbers: {preview}{suffix}"
        )
    return selected


_CONTIG_TOKEN_RE = re.compile(
    r"^(?:\d+(?:-\d+)?|[A-Za-z]\d+(?:-\d+)?|0)$"
)


def validate_contig(contig: str) -> None:
    """Validate documented RFdiffusion contig tokens conservatively."""

    value = contig.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    if not value:
        raise StructureValidationError("RFdiffusion contig cannot be empty")
    # RFdiffusion uses `/0 ` (including the space) as a chain break.
    value = value.replace("/0 ", "/0/")
    tokens = value.split("/")
    for token in tokens:
        token = token.strip()
        if not _CONTIG_TOKEN_RE.fullmatch(token):
            raise StructureValidationError(
                f"Invalid RFdiffusion contig token {token!r} in {contig!r}"
            )
        if "-" in token:
            numeric = token[1:] if token[0].isalpha() else token
            start, end = (int(part) for part in numeric.split("-", 1))
            if end < start:
                raise StructureValidationError(f"Reversed contig range {token!r}")


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
_EXTENDED_ALPHABET = AMINO_ACIDS | {"X", "B", "Z", "J", "U", "O"}


@dataclass(frozen=True)
class SequenceRecord:
    """One FASTA record, with complex chains split on ColabFold's `:`."""

    name: str
    chains: tuple[str, ...]

    @property
    def length(self) -> int:
        return sum(len(chain) for chain in self.chains)


def parse_fasta(path: Path) -> tuple[SequenceRecord, ...]:
    """Parse a FASTA file without reordering or rewriting any sequence."""

    if not path.is_file():
        raise StructureValidationError(f"Sequence file does not exist: {path}")
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise StructureValidationError(f"Cannot read sequence file {path}: {exc}") from exc

    records: list[SequenceRecord] = []
    name: str | None = None
    parts: list[str] = []

    def flush(line_number: int) -> None:
        if name is None:
            return
        sequence = "".join(parts).replace(" ", "").upper()
        if not sequence:
            raise StructureValidationError(
                f"Record {name!r} in {path.name} has no sequence (line {line_number})"
            )
        chains = tuple(sequence.split(":"))
        if any(not chain for chain in chains):
            raise StructureValidationError(
                f"Record {name!r} in {path.name} has an empty chain; "
                "':' separates the chains of a complex"
            )
        unknown = {character for chain in chains for character in chain} - _EXTENDED_ALPHABET
        if unknown:
            raise StructureValidationError(
                f"Record {name!r} in {path.name} contains non-amino-acid characters: "
                + "".join(sorted(unknown))
            )
        records.append(SequenceRecord(name=name, chains=chains))

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            flush(line_number)
            name = stripped[1:].strip()
            if not name:
                raise StructureValidationError(
                    f"Unnamed FASTA record at {path}:{line_number}"
                )
            parts = []
            continue
        if name is None:
            raise StructureValidationError(
                f"Sequence before the first '>' header at {path}:{line_number}"
            )
        parts.append(stripped)
    flush(len(lines))

    if not records:
        raise StructureValidationError(f"No FASTA records found in {path}")
    return tuple(records)


_MAP_MAGIC_OFFSET = 208
_LITTLE_ENDIAN_STAMP = b"\x44\x44"
_BIG_ENDIAN_STAMP = b"\x11\x11"


@dataclass(frozen=True)
class MapHeader:
    """The parts of an MRC/CCP4 header worth checking before a long run."""

    path: Path
    columns: int
    rows: int
    sections: int
    mode: int
    compressed: bool

    @property
    def voxels(self) -> int:
        return self.columns * self.rows * self.sections

    def describe(self) -> str:
        grid = f"{self.columns} x {self.rows} x {self.sections}"
        return f"{grid} voxels" + (", gzip compressed" if self.compressed else "")


def parse_map_header(path: Path) -> MapHeader:
    """Read an MRC/CCP4 density map header, transparently handling gzip."""

    if not path.is_file():
        raise StructureValidationError(f"Density map does not exist: {path}")
    compressed = path.suffix.lower() == ".gz"
    try:
        if compressed:
            import gzip

            with gzip.open(path, "rb") as handle:
                header = handle.read(1024)
        else:
            with path.open("rb") as handle:
                header = handle.read(1024)
    except (OSError, EOFError) as exc:
        raise StructureValidationError(f"Cannot read density map {path}: {exc}") from exc

    if len(header) < 1024:
        raise StructureValidationError(
            f"Density map {path.name} is truncated: its header is only {len(header)} bytes"
        )
    if header[_MAP_MAGIC_OFFSET : _MAP_MAGIC_OFFSET + 4] != b"MAP ":
        raise StructureValidationError(
            f"{path.name} is not an MRC/CCP4 density map: the 'MAP ' stamp is missing. "
            "CryoZeta reads .map, .mrc, and .map.gz files"
        )
    stamp = header[212:214]
    order = ">" if stamp == _BIG_ENDIAN_STAMP else "<"
    if stamp not in (_LITTLE_ENDIAN_STAMP, _BIG_ENDIAN_STAMP):
        order = "<"
    import struct

    columns, rows, sections, mode = struct.unpack(f"{order}4i", header[0:16])
    if min(columns, rows, sections) <= 0:
        raise StructureValidationError(
            f"Density map {path.name} reports an impossible grid: "
            f"{columns} x {rows} x {sections}"
        )
    return MapHeader(
        path=path,
        columns=columns,
        rows=rows,
        sections=sections,
        mode=mode,
        compressed=compressed,
    )


_NUCLEIC_ONLY = set("ACGTU")


def classify_sequence(sequence: str) -> str:
    """Return 'protein', 'dna', 'rna', or 'ambiguous' for a one-letter sequence.

    A, C, G and T are all valid amino acids as well as nucleotides, so a
    sequence written only from those letters cannot be classified safely and the
    caller has to be told to say which it is.
    """

    letters = set(sequence.upper())
    if not letters:
        return "ambiguous"
    if not letters <= _NUCLEIC_ONLY:
        return "protein"
    if "U" in letters and "T" not in letters:
        return "rna"
    return "ambiguous"
