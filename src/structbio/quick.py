"""Short positional commands, in the style of the SBGrid tool commands.

Each function here turns a handful of positional arguments into the same
configuration mapping that a YAML file would provide, so a quick command and a
YAML file take exactly the same validation and execution path. Anything that
cannot be expressed in three or four arguments stays in YAML.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from structbio.validation import (
    ResidueId,
    StructureValidationError,
    parse_pdb,
)


_SYMMETRY_RE = re.compile(r"^(?:c(\d+)|d(\d+)|tetrahedral)$", re.IGNORECASE)
_HOTSPOT_RE = re.compile(r"^[A-Za-z]:?-?\d+$")
_GPU_RE = re.compile(r"^\d+(?:,\d+)*$")


def split_list(value: str | None) -> list[str]:
    """Split a comma or space separated option value."""

    if not value:
        return []
    return [item for item in re.split(r"[,\s]+", value.strip()) if item]


def parse_gpu_ids(value: str) -> list[int]:
    if not _GPU_RE.fullmatch(value.strip()):
        raise ValueError(f"Invalid GPU selection {value!r}; expected digits such as 0 or 0,1")
    return [int(item) for item in value.strip().split(",")]


def symmetry_order(symmetry: str) -> int:
    """Return how many subunits a supported symmetry group generates."""

    match = _SYMMETRY_RE.fullmatch(symmetry.strip())
    if not match:
        raise ValueError(
            f"Unsupported symmetry {symmetry!r}; use cN, dN, or tetrahedral "
            "as documented by RFdiffusion"
        )
    if match.group(1):
        return int(match.group(1))
    if match.group(2):
        return 2 * int(match.group(2))
    return 12


def contiguous_segments(residues: tuple[ResidueId, ...]) -> list[tuple[int, int]]:
    """Group residue numbers, as they appear in the file, into closed ranges."""

    numbers = sorted({residue.number for residue in residues})
    segments: list[tuple[int, int]] = []
    start = previous = numbers[0]
    for number in numbers[1:]:
        if number == previous + 1:
            previous = number
            continue
        segments.append((start, previous))
        start = previous = number
    segments.append((start, previous))
    return segments


def _chain_for_target(pdb: Path, chain: str | None) -> tuple[str, tuple[ResidueId, ...]]:
    structure = parse_pdb(pdb)
    chains = sorted(structure.chains)
    if chain is None:
        if len(chains) != 1:
            raise StructureValidationError(
                f"{pdb.name} contains chains {' '.join(chains)}; "
                "name the target chain with --chain"
            )
        chain = chains[0]
    if chain not in structure.chains:
        raise StructureValidationError(
            f"Chain {chain!r} is absent from {pdb.name}; present chains: {' '.join(chains)}"
        )
    residues = structure.for_chain(chain)
    if any(residue.insertion_code for residue in residues):
        raise StructureValidationError(
            f"Chain {chain} of {pdb.name} uses insertion codes; write contigs in a YAML "
            "configuration instead of using the quick binder command"
        )
    return chain, residues


def binder_contig(pdb: Path, chain: str | None, binder_length: int) -> str:
    """Build a binder contig from the residue numbering actually in the PDB."""

    chain, residues = _chain_for_target(pdb, chain)
    segments = "/".join(f"{chain}{start}-{end}" for start, end in contiguous_segments(residues))
    return f"{segments}/0 {binder_length}-{binder_length}"


def full_length_contig(pdb: Path) -> str:
    """Build a contig covering every chain of an input structure at its own length."""

    structure = parse_pdb(pdb)
    lengths = [len(structure.for_chain(chain)) for chain in sorted(structure.chains)]
    return "/0 ".join(f"{length}-{length}" for length in lengths)


def _resources(gpus: int = 1) -> dict[str, Any]:
    return {"gpus": gpus}


def rfdiffusion_monomer(*, name: str, length: int, num_designs: int) -> dict[str, Any]:
    return {
        "tool": "rfdiffusion",
        "experiment": {"name": name},
        "design": {"mode": "monomer", "length": length, "num_designs": num_designs},
        "resources": _resources(),
    }


def rfdiffusion_symmetry(
    *, name: str, symmetry: str, length: int, num_designs: int
) -> dict[str, Any]:
    order = symmetry_order(symmetry)
    if length % order:
        raise ValueError(
            f"Total length {length} is not divisible by the {symmetry} subunit count {order}; "
            f"try {length - length % order} or {length + order - length % order}"
        )
    return {
        "tool": "rfdiffusion",
        "experiment": {"name": name},
        "design": {
            "mode": "symmetry",
            "symmetry": symmetry.lower(),
            "length": length,
            "num_designs": num_designs,
        },
        "resources": _resources(),
    }


def rfdiffusion_binder(
    *,
    name: str,
    target: Path,
    length: int,
    num_designs: int,
    chain: str | None = None,
    hotspots: str | None = None,
) -> dict[str, Any]:
    residues = split_list(hotspots)
    invalid = [item for item in residues if not _HOTSPOT_RE.fullmatch(item)]
    if invalid:
        raise ValueError(
            f"Invalid hotspot residue(s) {' '.join(invalid)}; use chain and number, such as B30"
        )
    return {
        "tool": "rfdiffusion",
        "experiment": {"name": name},
        "input": {"pdb": str(target)},
        "design": {
            "mode": "binder",
            "contigs": [binder_contig(target, chain, length)],
            "hotspot_residues": residues,
            "num_designs": num_designs,
        },
        "resources": _resources(),
    }


def rfdiffusion_partial(
    *, name: str, pdb: Path, steps: int, num_designs: int
) -> dict[str, Any]:
    return {
        "tool": "rfdiffusion",
        "experiment": {"name": name},
        "input": {"pdb": str(pdb)},
        "design": {
            "mode": "partial",
            "contigs": [full_length_contig(pdb)],
            "num_designs": num_designs,
        },
        "diffusion": {"partial_t": steps},
        "resources": _resources(),
    }


def proteinmpnn_design(
    *,
    name: str,
    input_path: Path,
    num_sequences: int,
    chains: str | None = None,
    designable: str | None = None,
    fixed: str | None = None,
    temperature: str | None = None,
    seed: int = 0,
    soluble: bool = False,
) -> dict[str, Any]:
    source = "directory" if input_path.is_dir() else "pdb"
    temperatures = [float(item) for item in split_list(temperature)] or [0.1]
    return {
        "tool": "proteinmpnn",
        "experiment": {"name": name},
        "input": {source: str(input_path)},
        "design": {
            "chains": split_list(chains),
            "designable_positions": split_list(designable),
            "fixed_positions": split_list(fixed),
            "num_sequences": num_sequences,
            "temperatures": temperatures,
            "seed": seed,
            "soluble_model": soluble,
        },
        "resources": _resources(),
    }


def _cryozeta_common(
    *, name: str, mode: str, large: bool, registration: str, gpu_ids: str | None
) -> dict[str, Any]:
    fragment: dict[str, Any] = {
        "tool": "cryozeta",
        "experiment": {"name": name},
        "mode": mode,
        "large": large,
        "gpu_ids": parse_gpu_ids(gpu_ids) if gpu_ids else [],
        "resources": _resources(),
    }
    if large:
        fragment["registration"] = registration
    return fragment


def cryozeta_predict(
    *,
    name: str,
    density_map: Path,
    sequences: Path,
    resolution: float,
    contour: float,
    mode: str = "combined",
    large: bool = False,
    registration: str = "auto",
    dna: str | None = None,
    rna: str | None = None,
    protein: str | None = None,
    msa_dir: Path | None = None,
    pairing_db: str | None = None,
    gpu_ids: str | None = None,
) -> dict[str, Any]:
    """Describe a CryoZeta target from a map and a FASTA of its chains."""

    fragment = _cryozeta_common(
        name=name, mode=mode, large=large, registration=registration, gpu_ids=gpu_ids
    )
    source: dict[str, Any] = {
        "map": str(density_map),
        "sequences": str(sequences),
        "resolution": resolution,
        "contour_level": contour,
    }
    chains = {
        "protein": split_list(protein),
        "dna": split_list(dna),
        "rna": split_list(rna),
    }
    if any(chains.values()):
        source["chains"] = chains
    msa: dict[str, Any] = {}
    if msa_dir is not None:
        msa["precomputed_msa_dir"] = str(msa_dir)
    if pairing_db:
        msa["pairing_db"] = pairing_db
    if msa:
        source["msa"] = msa
    fragment["input"] = source
    return fragment


def cryozeta_predict_json(
    *,
    name: str,
    input_json: Path,
    mode: str = "combined",
    large: bool = False,
    registration: str = "auto",
    gpu_ids: str | None = None,
) -> dict[str, Any]:
    """Run a CryoZeta target list that was written by hand."""

    fragment = _cryozeta_common(
        name=name, mode=mode, large=large, registration=registration, gpu_ids=gpu_ids
    )
    fragment["input"] = {"json": str(input_json)}
    return fragment


def colabfold_predict(
    *,
    name: str,
    sequences: Path,
    num_models: int = 5,
    msa_mode: str = "mmseqs2_uniref_env",
    templates: bool = False,
    relax: int = 0,
    num_recycle: int | None = None,
) -> dict[str, Any]:
    fragment: dict[str, Any] = {
        "tool": "colabfold",
        "experiment": {"name": name},
        "input": {"sequences": str(sequences)},
        "msa": {"mode": msa_mode, "templates": templates},
        "prediction": {"num_models": num_models},
        "relax": {"num_relax": relax, "use_gpu": bool(relax)},
        "resources": _resources(),
    }
    if num_recycle is not None:
        fragment["prediction"]["num_recycle"] = num_recycle
    return fragment
