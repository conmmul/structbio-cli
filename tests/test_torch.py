from pathlib import Path

import pytest

from structbio import environment
from structbio.environment import (
    TorchInstall,
    diagnose_torch,
    find_torch,
    select_torch_build,
    torch_install_command,
)


CUDA_VERSION_FILE = """from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.3.1+cu121'
debug = False
cuda: Optional[str] = '12.1'
git_version = 'abcdef'
hip: Optional[str] = None
"""

CPU_VERSION_FILE = CUDA_VERSION_FILE.replace("cuda: Optional[str] = '12.1'", "cuda: Optional[str] = None")


def _env_with_torch(root: Path, text: str) -> Path:
    site = root / "lib" / "python3.11" / "site-packages" / "torch"
    site.mkdir(parents=True)
    (site / "version.py").write_text(text)
    return root


@pytest.mark.parametrize(
    ("driver", "expected"),
    [
        ((11, 8), "cu118"),
        ((12, 0), "cu118"),  # cu121 needs 12.1, so step down rather than break
        ((12, 4), "cu124"),
        ((12, 9), "cu129"),
        ((13, 5), "cu132"),  # newest published build at or below the driver
        ((10, 2), "cpu"),  # older than any build we offer
        (None, "cpu"),
    ],
)
def test_the_build_never_exceeds_the_driver(driver, expected: str) -> None:
    assert select_torch_build(driver) == expected


def test_the_install_command_targets_the_right_environment_and_index() -> None:
    assert torch_install_command("mlfold", "cu124") == [
        "conda", "run", "-n", "mlfold",
        "pip", "install", "torch",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ]


def test_torch_is_read_without_importing_it(tmp_path: Path) -> None:
    found = find_torch(_env_with_torch(tmp_path / "cuda_env", CUDA_VERSION_FILE))
    assert found == TorchInstall(version="2.3.1+cu121", cuda="12.1")

    cpu = find_torch(_env_with_torch(tmp_path / "cpu_env", CPU_VERSION_FILE))
    assert cpu is not None and cpu.cpu_only

    assert find_torch(tmp_path / "empty") is None


def test_missing_torch_is_reported_with_a_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(environment, "conda_environments", lambda: {"mlfold": tmp_path})
    monkeypatch.setattr(environment, "driver_cuda_version", lambda: (12, 4))
    monkeypatch.setattr(environment, "detect_gpu", lambda: {"available": True, "models": [], "cuda_driver": "12.4"})

    problems, warnings, remedies = diagnose_torch("mlfold")
    assert any("PyTorch is not installed" in problem for problem in problems)
    assert warnings == []
    assert any("whl/cu124" in remedy for remedy in remedies)


def test_a_cpu_build_on_a_gpu_machine_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = _env_with_torch(tmp_path / "env", CPU_VERSION_FILE)
    monkeypatch.setattr(environment, "conda_environments", lambda: {"mlfold": prefix})
    monkeypatch.setattr(environment, "driver_cuda_version", lambda: (12, 4))
    monkeypatch.setattr(environment, "detect_gpu", lambda: {"available": True, "models": [], "cuda_driver": "12.4"})

    problems, warnings, _ = diagnose_torch("mlfold")
    # Slow is not the same as broken: this must not stop a run.
    assert problems == []
    assert any("CPU-only build" in warning for warning in warnings)


def test_a_cpu_build_is_fine_without_a_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = _env_with_torch(tmp_path / "env", CPU_VERSION_FILE)
    monkeypatch.setattr(environment, "conda_environments", lambda: {"mlfold": prefix})
    monkeypatch.setattr(environment, "driver_cuda_version", lambda: None)
    monkeypatch.setattr(environment, "detect_gpu", lambda: {"available": False, "models": [], "cuda_driver": None})

    problems, warnings, _ = diagnose_torch("mlfold")
    assert (problems, warnings) == ([], [])


def test_a_build_newer_than_the_driver_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wheel built for a newer CUDA than the driver supports will not run."""

    prefix = _env_with_torch(tmp_path / "env", CUDA_VERSION_FILE)  # built for 12.1
    monkeypatch.setattr(environment, "conda_environments", lambda: {"mlfold": prefix})
    monkeypatch.setattr(environment, "driver_cuda_version", lambda: (11, 8))
    monkeypatch.setattr(environment, "detect_gpu", lambda: {"available": True, "models": [], "cuda_driver": "11.8"})

    problems, _, remedies = diagnose_torch("mlfold")
    # A wheel the driver cannot run does stop a run.
    assert any("built for CUDA 12.1" in problem for problem in problems)
    assert any("whl/cu118" in remedy for remedy in remedies)


def test_a_matching_build_reports_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = _env_with_torch(tmp_path / "env", CUDA_VERSION_FILE)
    monkeypatch.setattr(environment, "conda_environments", lambda: {"mlfold": prefix})
    monkeypatch.setattr(environment, "driver_cuda_version", lambda: (12, 4))
    monkeypatch.setattr(environment, "detect_gpu", lambda: {"available": True, "models": [], "cuda_driver": "12.4"})

    assert diagnose_torch("mlfold") == ([], [], [])


def test_an_unknown_environment_is_not_diagnosed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment, "conda_environments", dict)
    assert diagnose_torch("absent") == ([], [], [])


def test_a_pinned_environment_is_never_told_to_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upgrading torch inside RFdiffusion's SE3nv breaks SE3Transformer."""

    from structbio.tools.rfdiffusion import SE3NV

    prefix = _env_with_torch(tmp_path / "env", CPU_VERSION_FILE)
    monkeypatch.setattr(environment, "conda_environments", lambda: {"SE3nv": prefix})
    monkeypatch.setattr(environment, "driver_cuda_version", lambda: (13, 0))
    monkeypatch.setattr(
        environment,
        "detect_gpu",
        lambda: {"available": True, "models": [], "cuda_driver": "13.0"},
    )

    _, warnings, remedies = diagnose_torch("SE3nv", pinned=SE3NV)
    assert any("CPU-only build" in warning for warning in warnings)
    joined = " ".join(remedies)
    # It must not suggest the newest wheel, which is what broke this before.
    assert "download.pytorch.org" not in joined
    assert "cu130" not in joined
    assert "pytorch=1.9 cudatoolkit=11.1" in joined
    assert "SE3Transformer" in joined
