"""Fixture builders shared by the test modules."""

from __future__ import annotations

import struct
from pathlib import Path


def write_map(path: Path, columns: int = 64, rows: int = 64, sections: int = 64) -> Path:
    """Write a minimal but structurally valid MRC/CCP4 map header."""

    header = bytearray(1024)
    header[0:16] = struct.pack("<4i", columns, rows, sections, 2)
    header[208:212] = b"MAP "
    header[212:216] = b"\x44\x44\x00\x00"
    path.write_bytes(bytes(header))
    return path


def write_fasta(path: Path, text: str = ">chain_A\nMKTAYIAKQRQISFVKSHFSRQ\n") -> Path:
    path.write_text(text)
    return path
