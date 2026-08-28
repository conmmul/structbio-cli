from pathlib import Path

import pytest

from helpers import write_map


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_pdb(fixture_dir: Path) -> Path:
    return (fixture_dir / "tiny.pdb").resolve()


@pytest.fixture
def tiny_map(tmp_path: Path) -> Path:
    return write_map(tmp_path / "target.map")
