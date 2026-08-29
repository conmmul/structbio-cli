from pathlib import Path

import pytest

from helpers import write_map


@pytest.fixture(autouse=True)
def isolated_workstation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never read or write the real workstation while the tests run.

    Every test gets its own home and configuration. A test that needs a
    particular file points the same variables somewhere else; setting them
    again wins over these defaults.
    """

    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(home / ".config/structbio/config.yaml"))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "no-lab-config.yaml"))
    # Finding a tool on disk mid-run is a convenience for researchers, not
    # something a test should get by accident.
    monkeypatch.setenv("STRUCTBIO_NO_AUTOCONFIG", "1")


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_pdb(fixture_dir: Path) -> Path:
    return (fixture_dir / "tiny.pdb").resolve()


@pytest.fixture
def tiny_map(tmp_path: Path) -> Path:
    return write_map(tmp_path / "target.map")
