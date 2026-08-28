from pathlib import Path

import pytest

from structbio import environment


def _fake_smi(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "nvidia-smi"
    script.write_text("#!/usr/bin/env bash\n" + body)
    script.chmod(0o755)
    return script


WORKING = '''case "$*" in
  *compute_cap*) echo "8.9"; echo "8.9";;
  *name,driver_version*) echo "NVIDIA L40S, 580.65.06"; echo "NVIDIA L40S, 580.65.06";;
  *) echo "CUDA Version: 13.0";;
esac
'''

OLD_DRIVER = '''case "$*" in
  *compute_cap*) echo "Invalid query field" >&2; exit 6;;
  *name,driver_version*) echo "NVIDIA RTX A6000, 470.82.01";;
  *) echo "CUDA Version: 11.4";;
esac
'''


def test_a_working_driver_is_read_completely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRUCTBIO_NVIDIA_SMI", str(_fake_smi(tmp_path, WORKING)))
    report = environment.gpu_report()
    assert report.available
    assert report.names == ("NVIDIA L40S", "NVIDIA L40S")
    assert report.capabilities == ((8, 9), (8, 9))
    assert report.driver_version == "580.65.06"
    assert report.driver_cuda == "13.0"
    assert report.capability_source == "nvidia-smi"
    assert report.error is None


def test_an_old_driver_falls_back_to_the_model_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """compute_cap needs a recent driver; an old one is not an absent GPU."""

    monkeypatch.setenv("STRUCTBIO_NVIDIA_SMI", str(_fake_smi(tmp_path, OLD_DRIVER)))
    report = environment.gpu_report()
    assert report.available
    assert report.capabilities == ((8, 6),)
    assert report.capability_source == "model name"
    assert report.error is None


def test_a_failing_driver_keeps_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "STRUCTBIO_NVIDIA_SMI",
        str(_fake_smi(tmp_path, 'echo "device handle error" >&2\nexit 255\n')),
    )
    report = environment.gpu_report()
    assert not report.available
    assert "exited with code 255" in report.error
    assert "device handle error" in report.error


def test_a_hanging_driver_times_out_with_an_explanation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRUCTBIO_NVIDIA_SMI", str(_fake_smi(tmp_path, "sleep 30\n")))
    monkeypatch.setattr(environment, "NVIDIA_SMI_TIMEOUT", 1.0)
    report = environment.gpu_report()
    assert not report.available
    assert "did not answer within" in report.error


def test_a_missing_nvidia_smi_says_where_it_looked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("STRUCTBIO_NVIDIA_SMI", raising=False)
    monkeypatch.setattr(environment.shutil, "which", lambda name: None)
    monkeypatch.setattr(environment, "NVIDIA_SMI_LOCATIONS", (str(tmp_path / "absent"),))
    report = environment.gpu_report()
    assert report.executable is None
    assert "STRUCTBIO_NVIDIA_SMI" in report.error


def test_the_override_is_used_even_when_one_is_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = _fake_smi(tmp_path, WORKING)
    monkeypatch.setenv("STRUCTBIO_NVIDIA_SMI", str(override))
    monkeypatch.setattr(environment.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    assert environment.nvidia_smi() == str(override)


def test_common_lab_cards_are_recognised_by_name() -> None:
    for name, expected in (
        ("NVIDIA A100-SXM4-80GB", (8, 0)),
        ("NVIDIA RTX A6000", (8, 6)),
        ("NVIDIA GeForce RTX 3090", (8, 6)),
        ("NVIDIA L40S", (8, 9)),
        ("NVIDIA GeForce RTX 4090", (8, 9)),
        ("NVIDIA H100 PCIe", (9, 0)),
        ("Tesla V100-SXM2-16GB", (7, 0)),
    ):
        assert environment.capability_from_name(name) == expected, name
    assert environment.capability_from_name("Some Future Card") is None


def test_a_present_but_unidentified_gpu_is_not_called_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Telling someone looking at a GPU that they have none helps nobody."""

    from structbio import provision
    from structbio.config import ToolInstallation

    unknown = '''case "$*" in
  *compute_cap*) exit 6;;
  *name,driver_version*) echo "NVIDIA Future Card, 999.0";;
  *) echo "CUDA Version: 14.0";;
esac
'''
    monkeypatch.setenv("STRUCTBIO_NVIDIA_SMI", str(_fake_smi(tmp_path, unknown)))
    checkout = tmp_path / "RFdiffusion"
    checkout.mkdir()
    plan = provision.plan_environment(
        "rfdiffusion",
        ToolInstallation(path=checkout, manager="conda", environment="SE3nv"),
    )
    assert not plan.possible
    assert "compute capability could not be determined" in plan.blocked
    assert any("--capability" in alternative for alternative in plan.alternatives)

    # Stating it directly gets a plan.
    stated = provision.plan_environment(
        "rfdiffusion",
        ToolInstallation(path=checkout, manager="conda", environment="SE3nv"),
        capability=(8, 9),
    )
    assert stated.possible
