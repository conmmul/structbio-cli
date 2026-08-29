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


@pytest.mark.parametrize("tool", ["rfdiffusion", "proteinmpnn"])
def test_the_probe_is_valid_python(tool: str) -> None:
    """It is generated source; it has to compile before it can be run anywhere."""

    compile(provision.probe_source(tool), "<probe>", "exec")


@pytest.mark.parametrize("tool", ["rfdiffusion", "proteinmpnn"])
def test_the_probe_runs_and_reports(tool: str) -> None:
    """Run it for real, so a broken generator cannot pass unnoticed again."""

    import subprocess
    import sys

    completed = subprocess.run(
        [sys.executable, "-c", provision.probe_source(tool)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    result = provision.parse_probe(completed.stdout)
    assert result.ok
    assert result.values["python"].startswith("3.")
    # Every module this tool needs must be asked about by name.
    for module in provision.PROBE_MODULES[tool]:
        assert module in result.values


def test_the_probe_asks_about_each_tools_own_modules() -> None:
    assert "dgl" in provision.probe_source("rfdiffusion")
    assert "se3_transformer" in provision.probe_source("rfdiffusion")
    assert "dgl" not in provision.probe_source("proteinmpnn")
    assert "numpy" in provision.probe_source("proteinmpnn")


def test_an_unknown_tool_still_produces_a_runnable_probe() -> None:
    compile(provision.probe_source("colabfold"), "<probe>", "exec")


def test_the_torch_pin_allows_any_patch_release() -> None:
    """A pinned patch release can be pruned from the index; the series is not."""

    assert provision.TORCH_RELEASE.endswith(".*")


def test_the_plan_checks_before_downloading_gigabytes(
    rfdiffusion: ToolInstallation, monkeypatch: pytest.MonkeyPatch
) -> None:
    _machine(monkeypatch, (8, 9))
    plan = provision.plan_environment("rfdiffusion", rfdiffusion)
    steps = [step.render() for step in plan.steps]
    dry = next(index for index, step in enumerate(steps) if "--dry-run" in step)
    install = next(
        index
        for index, step in enumerate(steps)
        if "torch==" in step and "--dry-run" not in step
    )
    assert dry < install


def test_a_python_outside_the_wheel_range_is_named(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        provision,
        "environment_facts",
        lambda name: {"python": "3.13.1", "tag": "cp313", "platform": "linux-x86_64"},
    )
    lines = provision.explain_pip_failure("SE3nv", "ERROR: from versions: none")
    assert any("Python 3.13.1" in line for line in lines)
    assert any("cp313, which neither builds for" in line for line in lines)


def test_a_non_x86_platform_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provision,
        "environment_facts",
        lambda name: {"python": "3.10.4", "tag": "cp310", "platform": "linux-aarch64"},
    )
    lines = provision.explain_pip_failure("SE3nv", "No matching distribution found")
    assert any("x86_64 only" in line for line in lines)
    assert any("ARM" in line for line in lines)


def test_unrelated_failures_are_not_explained_away(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provision, "environment_facts", lambda name: {})
    assert provision.explain_pip_failure("SE3nv", "Connection timed out") == []


def test_a_supported_environment_still_gets_the_generic_note(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        provision,
        "environment_facts",
        lambda name: {"python": "3.10.4", "tag": "cp310", "platform": "linux-x86_64"},
    )
    lines = provision.explain_pip_failure("SE3nv", "from versions: none")
    # This environment IS published for, so the environment must not be blamed.
    assert any("ARE published for" in line for line in lines)
    assert not any("ARM" in line or "neither builds for" in line for line in lines)


def test_an_unsupported_python_is_caught_before_downloading(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conda will build an environment on a Python PyTorch does not publish for."""

    monkeypatch.setattr(
        provision,
        "environment_facts",
        lambda name: {"python": "3.14.6", "tag": "cp314", "platform": "linux-x86_64"},
    )
    message = provision.unusable_python("SE3nv")
    assert message is not None
    assert "3.14.6" in message
    assert "Nothing was downloaded" in message
    assert "conda env remove -n SE3nv" in message


def test_a_supported_python_passes_silently(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provision,
        "environment_facts",
        lambda name: {"python": "3.10.14", "tag": "cp310", "platform": "linux-x86_64"},
    )
    assert provision.unusable_python("SE3nv") is None


def test_an_unreadable_environment_is_not_condemned(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provision, "environment_facts", lambda name: {})
    assert provision.unusable_python("SE3nv") is None


def test_the_environments_own_interpreter_is_used(tmp_path: Path, monkeypatch) -> None:
    """conda run resolves through PATH and can answer with the base Python."""

    prefix = tmp_path / "envs" / "SE3nv"
    (prefix / "bin").mkdir(parents=True)
    interpreter = prefix / "bin" / "python"
    interpreter.touch()
    monkeypatch.setattr(
        provision.environment, "conda_environments", lambda: {"SE3nv": prefix}
    )
    assert provision.environment_python("SE3nv") == interpreter


def test_an_environment_without_an_interpreter_is_reported_as_such(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty prefix means the creation step did not finish."""

    prefix = tmp_path / "envs" / "SE3nv"
    prefix.mkdir(parents=True)
    monkeypatch.setattr(
        provision.environment, "conda_environments", lambda: {"SE3nv": prefix}
    )
    assert provision.environment_python("SE3nv") is None
    facts = provision.environment_facts("SE3nv")
    assert facts["missing_python"] == str(prefix)

    message = provision.unusable_python("SE3nv")
    assert message is not None
    assert "contains no Python interpreter" in message
    assert "did not finish" in message


def test_the_real_interpreter_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the facts probe against a genuine interpreter, end to end."""

    import sys

    prefix = tmp_path / "envs" / "real"
    (prefix / "bin").mkdir(parents=True)
    link = prefix / "bin" / "python"
    link.symlink_to(sys.executable)
    monkeypatch.setattr(
        provision.environment, "conda_environments", lambda: {"real": prefix}
    )
    facts = provision.environment_facts("real")
    assert facts["python"].startswith("3.")
    assert facts["tag"].startswith("cp3")
    assert "x86_64" in facts["platform"] or "arm64" in facts["platform"]


def test_the_reported_python_path_is_included_in_the_complaint(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        provision,
        "environment_facts",
        lambda name: {
            "python": "3.14.6",
            "tag": "cp314",
            "platform": "linux-x86_64",
            "executable": "/opt/anaconda3/bin/python",
        },
    )
    message = provision.unusable_python("SE3nv")
    assert "/opt/anaconda3/bin/python" in message
    assert "not inside that environment" in message


def test_a_backup_name_never_collides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provision.environment,
        "conda_environments",
        lambda: {"SE3nv": Path("/a"), "SE3nv-before-1": Path("/b")},
    )
    assert provision.backup_name("SE3nv") == "SE3nv-before-2"


def test_an_environment_is_renamed_not_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A researcher's working environment is not structbio's to destroy."""

    calls: list[list[str]] = []

    class _Ok:
        returncode = 0

    monkeypatch.setattr(
        provision.environment, "conda_environments", lambda: {"SE3nv": Path("/a")}
    )
    monkeypatch.setattr(
        provision.subprocess, "run", lambda argv, **k: calls.append(argv) or _Ok()
    )
    assert provision.move_aside("SE3nv") == "SE3nv-before-1"
    assert calls == [["conda", "rename", "-n", "SE3nv", "SE3nv-before-1"]]
    assert not any("remove" in " ".join(call) for call in calls)


def test_a_failed_rename_is_reported_rather_than_forced(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Failed:
        returncode = 1

    monkeypatch.setattr(
        provision.environment, "conda_environments", lambda: {"SE3nv": Path("/a")}
    )
    monkeypatch.setattr(provision.subprocess, "run", lambda argv, **k: _Failed())
    assert provision.move_aside("SE3nv") is None


def test_a_reachable_index_is_not_blamed_on_the_environment(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """cp310 on x86_64 is published for, so 'none' means the index was not read."""

    monkeypatch.setattr(
        provision,
        "environment_facts",
        lambda name: {"python": "3.10.21", "tag": "cp310", "platform": "linux-x86_64"},
    )
    lines = provision.explain_pip_failure("SE3nv", "from versions: none")
    joined = " ".join(lines)
    assert "ARE published for" in joined
    assert "network" in joined
    assert "proxy" in joined
    # It must not claim the wheels are missing, which was the wrong answer.
    assert "neither builds for" not in joined
    assert "x86_64 only" not in joined


def test_repair_keeps_the_environment_and_only_changes_pytorch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from structbio.config import ToolInstallation

    checkout = tmp_path / "RFdiffusion"
    checkout.mkdir()
    monkeypatch.setattr(provision, "environment_exists", lambda name: True)
    monkeypatch.setattr(
        provision, "environment_facts", lambda name: {"python": "3.9.25", "tag": "cp39"}
    )
    monkeypatch.setattr(provision.environment, "gpu_capabilities", lambda: [(8, 9)])

    plan = provision.repair_plan(
        "rfdiffusion",
        ToolInstallation(path=checkout, manager="conda", environment="SE3nv"),
    )
    assert plan.possible
    rendered = [step.render() for step in plan.steps]
    # Nothing is created or removed.
    assert not any("conda create" in step or "env remove" in step for step in rendered)
    assert not any("rename" in step for step in rendered)
    # PyTorch and DGL are the verified conda pair for this Python.
    assert any("pytorch=2.3.1=py3.9_cuda11.8_cudnn8.7.0_0" in step for step in rendered)
    assert any("dgl=2.4.0.th23.cu118=py39_0" in step for step in rendered)
    # Installed from conda, not from the pip index this machine cannot reach.
    assert not any("download.pytorch.org" in step for step in rendered)


def test_repair_refuses_a_python_it_has_no_verified_pair_for(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    from structbio.config import ToolInstallation

    monkeypatch.setattr(provision, "environment_exists", lambda name: True)
    monkeypatch.setattr(
        provision, "environment_facts", lambda name: {"python": "3.12.1", "tag": "cp312"}
    )
    plan = provision.repair_plan(
        "rfdiffusion", ToolInstallation(manager="conda", environment="SE3nv")
    )
    assert not plan.possible
    assert "no verified conda pairing" in plan.blocked


def test_repair_needs_the_environment_to_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    from structbio.config import ToolInstallation

    monkeypatch.setattr(provision, "environment_exists", lambda name: False)
    plan = provision.repair_plan(
        "proteinmpnn", ToolInstallation(manager="conda", environment="mlfold")
    )
    assert not plan.possible
    assert "no conda environment named" in plan.blocked


def test_proteinmpnn_repair_does_not_touch_dgl(monkeypatch: pytest.MonkeyPatch) -> None:
    from structbio.config import ToolInstallation

    monkeypatch.setattr(provision, "environment_exists", lambda name: True)
    monkeypatch.setattr(
        provision, "environment_facts", lambda name: {"python": "3.9.25", "tag": "cp39"}
    )
    monkeypatch.setattr(provision.environment, "gpu_capabilities", lambda: [(8, 9)])
    plan = provision.repair_plan(
        "proteinmpnn", ToolInstallation(manager="conda", environment="mlfold")
    )
    rendered = " ".join(step.render() for step in plan.steps)
    assert "pytorch=2.3.1" in rendered
    # ProteinMPNN does not use DGL, so no dgl package is installed. The dglteam
    # channel appears in the channel list, so check the package spec instead.
    assert "dgl=" not in rendered
    assert len(plan.steps) == 1
