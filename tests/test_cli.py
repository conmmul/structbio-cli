import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from structbio import __version__
from structbio.cli import app


runner = CliRunner()


def _rf_config(tmp_path: Path) -> Path:
    config = tmp_path / "rf.yaml"
    config.write_text(
        "tool: rfdiffusion\n"
        "experiment: {name: dry}\n"
        "design: {mode: monomer, length: 100, num_designs: 1}\n"
        f"experiments_root: {tmp_path / 'experiments'}\n"
    )
    return config


def test_dry_run_executes_nothing_and_creates_nothing(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rfdiffusion", "run", str(_rf_config(tmp_path)), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "nothing was created or executed" in result.output
    assert not (tmp_path / "experiments").exists()


def test_command_generation_creates_nothing(tmp_path: Path) -> None:
    result = runner.invoke(app, ["rfdiffusion", "command", str(_rf_config(tmp_path))])
    assert result.exit_code == 0, result.output
    assert "inference.num_designs=1" in result.output
    assert not (tmp_path / "experiments").exists()


def test_submit_dry_run_does_not_call_slurm(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["rfdiffusion", "submit", str(_rf_config(tmp_path)), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "#SBATCH" in result.output
    assert "nothing was created, executed, or submitted" in result.output
    assert not (tmp_path / "experiments").exists()


def test_cli_override_has_highest_precedence(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "rfdiffusion",
            "command",
            str(_rf_config(tmp_path)),
            "--set",
            "design.num_designs=7",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "inference.num_designs=7" in result.output


def _workstation_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point structbio at a fake, non-executable tool checkout."""

    checkout = tmp_path / "software" / "RFdiffusion" / "scripts"
    checkout.mkdir(parents=True)
    (checkout / "run_inference.py").write_text("raise SystemExit('never executed')\n")
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  rfdiffusion:\n"
        f"    path: {tmp_path / 'software' / 'RFdiffusion'}\n"
        "    executable: scripts/run_inference.py\n"
        "    manager: none\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.chdir(tmp_path)
    return user_config


def test_quick_monomer_dry_run_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["rfdiffusion", "monomer", "150", "my_designs", "-n", "4", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "contigmap.contigs=[150-150]" in result.output
    assert "inference.num_designs=4" in result.output
    assert "my_designs/my_designs" in result.output
    assert "nothing was created or executed" in result.output
    assert not (tmp_path / "my_designs").exists()


def test_quick_command_names_output_files_after_the_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    result = runner.invoke(app, ["rfdiffusion", "monomer", "80", "nested/run_one", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "nested/run_one/run_one" in result.output


def test_quick_gpu_selection_is_passed_as_an_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["rfdiffusion", "monomer", "80", "out", "--gpu", "1,2", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "CUDA_VISIBLE_DEVICES=1,2" in result.output

    rejected = runner.invoke(
        app, ["rfdiffusion", "monomer", "80", "out", "--gpu", "cuda:0", "--dry-run"]
    )
    assert rejected.exit_code == 2
    assert "Invalid GPU selection" in rejected.output


def test_quick_run_refuses_a_non_empty_output_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    existing = tmp_path / "occupied"
    existing.mkdir()
    (existing / "previous.pdb").write_text("ATOM\n")
    result = runner.invoke(app, ["rfdiffusion", "monomer", "80", "occupied"])
    assert result.exit_code == 2
    assert "Refusing to write into the existing non-empty folder" in result.output
    assert (existing / "previous.pdb").read_text() == "ATOM\n"


def test_quick_run_stops_when_the_tool_is_not_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_config = tmp_path / "user.yaml"
    user_config.write_text("tools:\n  rfdiffusion:\n    manager: none\n")
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["rfdiffusion", "monomer", "80", "out"])
    assert result.exit_code == 2
    assert "not available on this machine" in result.output
    assert not (tmp_path / "out").exists()


def test_quick_binder_reports_a_bad_chain_before_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_pdb: Path
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    result = runner.invoke(
        app, ["rfdiffusion", "binder", str(tiny_pdb), "100", "out", "--dry-run"]
    )
    assert result.exit_code == 2
    assert "name the target chain with --chain" in result.output


def test_quick_binder_builds_the_contig_from_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_pdb: Path
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        [
            "rfdiffusion",
            "binder",
            str(tiny_pdb),
            "100",
            "binders",
            "--chain",
            "B",
            "--hotspots",
            "B11",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "contigmap.contigs=[B10-12/0 100-100]" in result.output
    assert "ppi.hotspot_res=[B11]" in result.output


def test_quick_run_writes_outputs_and_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    script = tmp_path / "software" / "RFdiffusion" / "scripts" / "run_inference.py"
    script.write_text(
        "import os, sys, pathlib\n"
        "prefix = [a.split('=', 1)[1] for a in sys.argv[1:]\n"
        "          if a.startswith('inference.output_prefix=')][0]\n"
        "pathlib.Path(prefix + '_0.pdb').write_text('ATOM\\n')\n"
        "pathlib.Path(prefix + '_gpu.txt').write_text(os.environ.get('CUDA_VISIBLE_DEVICES', ''))\n"
    )
    result = runner.invoke(
        app, ["rfdiffusion", "monomer", "40", "designs", "--gpu", "3"]
    )
    assert result.exit_code == 0, result.output
    output_dir = tmp_path / "designs"
    assert (output_dir / "designs_0.pdb").read_text() == "ATOM\n"
    assert (output_dir / "designs_gpu.txt").read_text() == "3"
    metadata = json.loads((output_dir / ".structbio" / "metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["tool"] == "rfdiffusion"

    status = runner.invoke(app, ["status", str(output_dir)])
    assert status.exit_code == 0, status.output
    assert "completed" in status.output


def test_setup_writes_a_configuration_and_tool_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config" / "structbio.yaml"
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(config_path))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    result = runner.invoke(app, ["setup", "--bin-dir", str(tmp_path / "bin")])
    assert result.exit_code == 0, result.output
    assert "tools:" in config_path.read_text()
    assert (tmp_path / "bin" / "rfdiffusion").stat().st_mode & 0o111

    config_path.write_text("tools: {}\n")
    again = runner.invoke(app, ["setup", "--bin-dir", str(tmp_path / "bin")])
    assert again.exit_code == 0, again.output
    assert config_path.read_text() == "tools: {}\n"


def test_version_flag_reports_the_package_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == __version__


def test_setup_says_loudly_when_the_commands_are_not_on_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.setenv("SHELL", "/bin/zsh")
    bin_dir = tmp_path / "bin"

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    missing = runner.invoke(app, ["setup", "--bin-dir", str(bin_dir)])
    assert missing.exit_code == 0, missing.output
    assert "will NOT work yet" in missing.output
    assert f'export PATH="{bin_dir.resolve()}:$PATH"' in missing.output
    assert "~/.zshrc" in missing.output

    monkeypatch.setenv("PATH", f"{bin_dir.resolve()}:/usr/bin:/bin")
    present = runner.invoke(app, ["setup", "--bin-dir", str(bin_dir)])
    assert present.exit_code == 0, present.output
    assert "will NOT work yet" not in present.output


def test_gpu_auto_picks_the_card_with_most_free_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    monkeypatch.setattr("structbio.environment.select_idle_gpu", lambda: 2)
    result = runner.invoke(
        app, ["rfdiffusion", "monomer", "80", "out", "--gpu", "auto", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "Using GPU 2" in result.output
    assert "CUDA_VISIBLE_DEVICES=2" in result.output


def test_gpu_auto_without_nvidia_smi_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workstation_config(tmp_path, monkeypatch)
    monkeypatch.setattr("structbio.environment.select_idle_gpu", lambda: None)
    result = runner.invoke(
        app, ["rfdiffusion", "monomer", "80", "out", "--gpu", "auto", "--dry-run"]
    )
    assert result.exit_code == 2
    assert "needs nvidia-smi" in result.output


def test_colabfold_quick_command_folds_a_proteinmpnn_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n  colabfold:\n    executable: colabfold_batch\n    manager: none\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.chdir(tmp_path)
    designs = tmp_path / "my_sequences" / "seqs"
    designs.mkdir(parents=True)
    (designs / "design.fa").write_text(">design_0\nMKTAYIAKQRQISFVKSHFSRQ\n")

    result = runner.invoke(
        app, ["colabfold", "predict", "my_sequences", "my_folds", "-n", "2", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert f"colabfold_batch {designs}" in result.output
    assert "--num-models 2" in result.output
    assert "leaves this machine" in result.output


def _fake_install(root: Path) -> Path:
    (root / "ProteinMPNN").mkdir(parents=True)
    (root / "ProteinMPNN" / "protein_mpnn_run.py").touch()
    return root


def test_setup_writes_a_configuration_from_what_it_finds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from structbio import discovery

    software = _fake_install(tmp_path / "software")
    monkeypatch.setattr(discovery, "conda_environments", dict)
    scan = discovery.discover  # capture before patching, or the lambda calls itself
    monkeypatch.setattr(discovery, "discover", lambda **_: scan(roots=(str(software),)))
    config_path = tmp_path / "config" / "structbio.yaml"
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(config_path))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))

    result = runner.invoke(app, ["setup", "--bin-dir", str(tmp_path / "bin")])
    assert result.exit_code == 0, result.output
    assert "proteinmpnn    found" in result.output
    assert "rfdiffusion    not found" in result.output

    written = yaml.safe_load(config_path.read_text())
    assert written["tools"]["proteinmpnn"]["path"] == str(software / "ProteinMPNN")
    assert "rfdiffusion" not in written["tools"]


def test_setup_update_merges_and_keeps_a_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from structbio import discovery

    software = _fake_install(tmp_path / "software")
    monkeypatch.setattr(discovery, "conda_environments", dict)
    scan = discovery.discover  # capture before patching, or the lambda calls itself
    monkeypatch.setattr(discovery, "discover", lambda **_: scan(roots=(str(software),)))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("tools:\n  rfdiffusion:\n    path: /my/RFdiffusion\n")
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(config_path))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))

    unchanged = runner.invoke(app, ["setup", "--bin-dir", str(tmp_path / "bin")])
    assert "already exists" in unchanged.output
    assert config_path.read_text().startswith("tools:")

    updated = runner.invoke(app, ["setup", "--update", "--bin-dir", str(tmp_path / "bin")])
    assert updated.exit_code == 0, updated.output
    merged = yaml.safe_load(config_path.read_text())
    assert merged["tools"]["rfdiffusion"]["path"] == "/my/RFdiffusion"
    assert merged["tools"]["proteinmpnn"]["path"] == str(software / "ProteinMPNN")
    assert (tmp_path / "config.yaml.bak").read_text().startswith("tools:")


def test_detect_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(config_path))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.setattr("structbio.cli.discovery.discover", lambda **_: {})
    result = runner.invoke(app, ["detect"])
    assert result.exit_code == 0, result.output
    assert "Not found: rfdiffusion" in result.output
    assert not config_path.exists()


def test_install_dry_run_clones_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(tmp_path / "config.yaml"))
    result = runner.invoke(
        app, ["install", "cryozeta", "--into", str(tmp_path / "software"), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "non-commercial" in result.output
    assert "pixi run setup" in result.output
    assert "Dry run: nothing was cloned" in result.output
    assert not (tmp_path / "software").exists()


def test_install_refuses_an_existing_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(tmp_path / "config.yaml"))
    (tmp_path / "software" / "CryoZeta").mkdir(parents=True)
    result = runner.invoke(
        app,
        ["install", "cryozeta", "--into", str(tmp_path / "software"), "--yes"],
    )
    assert result.exit_code == 2
    assert "already exists" in result.output


def test_install_rejects_an_unknown_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(tmp_path / "config.yaml"))
    result = runner.invoke(app, ["install", "alphafold", "--dry-run"])
    assert result.exit_code == 2
    assert "known tools" in result.output


def test_an_unavailable_tool_explains_why_and_how_to_fix_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  rfdiffusion:\n"
        f"    path: {tmp_path / 'absent' / 'RFdiffusion'}\n"
        "    executable: scripts/run_inference.py\n"
        "    manager: none\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["rfdiffusion", "monomer", "80", "out"])
    assert result.exit_code == 2
    assert "the configured path does not exist" in result.output
    assert "To fix it:" in result.output
    assert "structbio install rfdiffusion" in result.output
    assert not (tmp_path / "out").exists()


def test_doctor_names_the_reason_a_tool_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "RFdiffusion"
    checkout.mkdir()  # present, but the entry-point script is missing
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  rfdiffusion:\n"
        f"    path: {checkout}\n"
        "    executable: scripts/run_inference.py\n"
        "    manager: none\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "CONFIGURED, UNAVAILABLE" in result.output
    assert "does not contain scripts/run_inference.py" in result.output
    assert "fix:" in result.output


def _conda_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch: str | None) -> Path:
    """A ProteinMPNN checkout plus a conda environment, with or without torch."""

    checkout = tmp_path / "ProteinMPNN"
    checkout.mkdir()
    (checkout / "protein_mpnn_run.py").touch()
    prefix = tmp_path / "envs" / "mlfold"
    prefix.mkdir(parents=True)
    if torch is not None:
        site = prefix / "lib" / "python3.11" / "site-packages" / "torch"
        site.mkdir(parents=True)
        (site / "version.py").write_text(torch)

    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  proteinmpnn:\n"
        f"    path: {checkout}\n"
        "    executable: protein_mpnn_run.py\n"
        "    manager: conda\n"
        "    environment: mlfold\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.setattr("structbio.environment.conda_environments", lambda: {"mlfold": prefix})
    monkeypatch.setattr("structbio.environment.shutil.which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        "structbio.environment.detect_gpu",
        lambda: {"available": True, "models": ["Test GPU"], "cuda_driver": "12.4"},
    )
    monkeypatch.chdir(tmp_path)
    return checkout


def test_a_run_stops_when_torch_is_missing_and_says_how_to_install_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_pdb: Path
) -> None:
    _conda_tool(tmp_path, monkeypatch, torch=None)
    result = runner.invoke(app, ["proteinmpnn", "design", str(tiny_pdb), "2", "out"])
    assert result.exit_code == 2
    assert "PyTorch is not installed" in result.output
    assert "whl/cu124" in result.output


def test_fix_env_prints_the_command_and_installs_nothing_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _conda_tool(tmp_path, monkeypatch, torch=None)
    calls: list[list[str]] = []
    monkeypatch.setattr("structbio.cli.subprocess.run", lambda argv, **_: calls.append(argv))

    result = runner.invoke(app, ["fix-env", "proteinmpnn"])
    assert result.exit_code == 0, result.output
    assert "Driver CUDA: 12.4" in result.output
    assert "PyTorch build to use: cu124" in result.output
    assert "conda run -n mlfold pip install torch" in result.output
    assert "Nothing was installed" in result.output
    assert calls == []


def test_fix_env_leaves_a_working_installation_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _conda_tool(
        tmp_path,
        monkeypatch,
        torch="__version__ = '2.3.1+cu121'\ncuda: Optional[str] = '12.1'\n",
    )
    result = runner.invoke(app, ["fix-env", "proteinmpnn"])
    assert result.exit_code == 0, result.output
    assert "PyTorch 2.3.1+cu121, built for CUDA 12.1" in result.output
    assert "pass --force to replace it" in result.output
    assert "pip install torch" not in result.output


def test_fix_env_rejects_an_unknown_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _conda_tool(tmp_path, monkeypatch, torch=None)
    result = runner.invoke(app, ["fix-env", "alphafold"])
    assert result.exit_code == 2
    assert "Unknown tool" in result.output
