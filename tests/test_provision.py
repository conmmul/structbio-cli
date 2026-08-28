from pathlib import Path

import pytest

from structbio import environment, provision
from structbio.config import ToolInstallation


@pytest.fixture
def rfdiffusion(tmp_path: Path) -> ToolInstallation:
    checkout = tmp_path / "RFdiffusion"
    (checkout / "env" / "SE3Transformer").mkdir(parents=True)
    (checkout / "scripts").mkdir()
    return ToolInstallation(
        path=checkout,
        executable="scripts/run_inference.py",
        manager="conda",
        environment="SE3nv",
    )


def _machine(monkeypatch: pytest.MonkeyPatch, capability, driver=(13, 0)) -> None:
    monkeypatch.setattr(environment, "gpu_capabilities", lambda: [capability] if capability else [])
    monkeypatch.setattr(environment, "driver_cuda_version", lambda: driver)


def test_an_ampere_card_gets_the_upstream_pinned_versions(
    rfdiffusion: ToolInstallation, monkeypatch: pytest.MonkeyPatch
) -> None:
    _machine(monkeypatch, (8, 6))
    plan = provision.plan_environment("rfdiffusion", rfdiffusion)
    assert plan.possible and plan.upstream_verified
    create = plan.steps[0].render()
    assert "pytorch=1.9" in create and "cudatoolkit=11.1" in create
    # The channel order is the fix for conda otherwise choosing a CPU build.
    assert create.index("-c pytorch") < create.index("-c conda-forge")


@pytest.mark.parametrize("capability", [(8, 9), (9, 0)])
def test_a_newer_card_gets_a_coherent_modern_pairing(
    rfdiffusion: ToolInstallation, monkeypatch: pytest.MonkeyPatch, capability
) -> None:
    _machine(monkeypatch, capability)
    plan = provision.plan_environment("rfdiffusion", rfdiffusion)
    assert plan.possible
    # Not an upstream combination, and it must say so.
    assert not plan.upstream_verified
    assert any("NOT one the RFdiffusion authors have published" in note for note in plan.notes)

    rendered = " ".join(step.render() for step in plan.steps)
    build = "cu118"
    assert f"torch=={provision.TORCH_RELEASE}" in rendered
    assert f"download.pytorch.org/whl/{build}" in rendered
    # PyTorch and DGL must be built for the same CUDA, or neither works.
    assert f"wheels/torch-{provision.DGL_TORCH_INDEX}/{build}/" in rendered


def test_a_card_newer_than_dgl_supports_is_refused_not_guessed(
    rfdiffusion: ToolInstallation, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blackwell needs CUDA 12.8; no PyTorch/DGL pairing publishes that."""

    _machine(monkeypatch, (12, 0))
    plan = provision.plan_environment("rfdiffusion", rfdiffusion)
    assert not plan.possible
    assert "no version combination that runs it on this card" in plan.blocked
    assert any("older card" in alternative for alternative in plan.alternatives)
    assert any("ProteinMPNN, ColabFold and" in a for a in plan.alternatives)


def test_a_build_is_never_newer_than_the_driver_can_load(
    rfdiffusion: ToolInstallation, monkeypatch: pytest.MonkeyPatch
) -> None:
    _machine(monkeypatch, (8, 9), driver=(11, 8))
    plan = provision.plan_environment("rfdiffusion", rfdiffusion)
    assert "download.pytorch.org/whl/cu118" in " ".join(s.render() for s in plan.steps)

    _machine(monkeypatch, (8, 9), driver=(11, 0))
    assert not provision.plan_environment("rfdiffusion", rfdiffusion).possible


def test_no_gpu_means_no_rfdiffusion_plan(
    rfdiffusion: ToolInstallation, monkeypatch: pytest.MonkeyPatch
) -> None:
    _machine(monkeypatch, None)
    plan = provision.plan_environment("rfdiffusion", rfdiffusion)
    assert not plan.possible
    assert "no NVIDIA GPU" in plan.blocked


def test_proteinmpnn_needs_no_hardware_specific_pinning(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    _machine(monkeypatch, (12, 0))
    plan = provision.plan_environment(
        "proteinmpnn", ToolInstallation(manager="conda", environment="mlfold")
    )
    assert plan.possible and plan.upstream_verified
    rendered = " ".join(step.render() for step in plan.steps)
    assert "conda create -y -n mlfold" in rendered
    assert "pip install torch --index-url" in rendered


def test_tools_that_manage_themselves_are_left_alone() -> None:
    for tool in ("colabfold", "cryozeta"):
        plan = provision.plan_environment(tool, ToolInstallation())
        assert not plan.possible
        assert "manages its own environment" in plan.blocked


def test_the_probe_reports_a_card_without_kernels() -> None:
    """is_available() is True in this case; only computing on the GPU catches it."""

    result = provision.parse_probe(
        'noise\nSTRUCTBIO_PROBE {"torch": "1.9.1", "torch_cuda": "11.1", '
        '"cuda_available": true, "device": "NVIDIA RTX 5090", '
        '"gpu_allocation": false, "gpu_error": "no kernel image is available"}\n'
    )
    assert result.ok
    failures = result.failures()
    assert any("cannot compute on it" in failure for failure in failures)
    assert any("no kernel image" in failure for failure in failures)


def test_the_probe_reports_a_working_environment() -> None:
    result = provision.parse_probe(
        'STRUCTBIO_PROBE {"torch": "2.3.1+cu121", "torch_cuda": "12.1", '
        '"cuda_available": true, "device": "NVIDIA L40S", "gpu_allocation": true, '
        '"dgl": true, "se3_transformer": true, "hydra": true}'
    )
    assert result.failures() == []
    assert "NVIDIA L40S" in result.summary()


def test_the_probe_reports_a_missing_module() -> None:
    result = provision.parse_probe(
        'STRUCTBIO_PROBE {"torch": "2.3.1+cu121", "torch_cuda": "12.1", '
        '"cuda_available": true, "gpu_allocation": true, "dgl": false, '
        '"dgl_error": "No module named dgl"}'
    )
    assert any("dgl could not be imported" in failure for failure in result.failures())


def test_unreadable_probe_output_is_not_mistaken_for_success() -> None:
    result = provision.parse_probe("conda: command not found")
    assert not result.ok
    assert result.failures()
