from pathlib import Path

import pytest


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_pdb(fixture_dir: Path) -> Path:
    return (fixture_dir / "tiny.pdb").resolve()
