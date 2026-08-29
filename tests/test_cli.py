import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from structbio import __version__
from structbio.cli import app
from structbio.config import load_settings
from structbio.tools import get_backends


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


def test_setup_puts_the_commands_on_path_by_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELL", "/bin/zsh")
    profile = Path.home() / ".zshrc"
    profile.write_text("# existing configuration\n")
    bin_dir = tmp_path / "bin"

    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    added = runner.invoke(app, ["setup", "--bin-dir", str(bin_dir)])
    assert added.exit_code == 0, added.output
    assert f'export PATH="{bin_dir.resolve()}:$PATH"' in profile.read_text()
    assert "# existing configuration" in profile.read_text()
    assert str(profile) in added.output

    # Running it again must not add the line a second time.
    again = runner.invoke(app, ["setup", "--bin-dir", str(bin_dir)])
    assert again.exit_code == 0, again.output
    assert profile.read_text().count("export PATH=") == 1

    monkeypatch.setenv("PATH", f"{bin_dir.resolve()}:/usr/bin:/bin")
    present = runner.invoke(app, ["setup", "--bin-dir", str(bin_dir)])
    assert "PATH              already includes" in present.output


def test_setup_leaves_an_unwritable_shell_profile_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A managed workstation can own the researcher's shell files."""

    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr("structbio.wrappers._writable", lambda path: False)
    bin_dir = tmp_path / "bin"

    result = runner.invoke(app, ["setup", "--bin-dir", str(bin_dir)])
    assert result.exit_code == 0, result.output
    assert "could not be set" in result.output
    assert f'export PATH="{bin_dir.resolve()}:$PATH"' in result.output


def test_setup_can_be_told_not_to_touch_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SHELL", "/bin/zsh")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    profile = Path.home() / ".zshrc"
    profile.write_text("# existing configuration\n")

    result = runner.invoke(app, ["setup", "--no-path", "--bin-dir", str(tmp_path / "bin")])
    assert result.exit_code == 0, result.output
    assert profile.read_text() == "# existing configuration\n"


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


def test_setup_merges_new_tools_and_keeps_a_backup(
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

    updated = runner.invoke(app, ["setup", "--bin-dir", str(tmp_path / "bin")])
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


def test_doctor_ends_with_what_can_and_cannot_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wall of status is only useful if it ends with the answer."""

    checkout = tmp_path / "RFdiffusion" / "scripts"
    checkout.mkdir(parents=True)
    (checkout / "run_inference.py").touch()
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  rfdiffusion:\n"
        f"    path: {tmp_path / 'RFdiffusion'}\n"
        "    executable: scripts/run_inference.py\n"
        "    manager: none\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))

    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "Ready to run: rfdiffusion" in result.output
    assert "Not usable yet:" in result.output
    assert "structbio setup" in result.output


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


def test_a_missing_torch_warns_without_stopping_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_pdb: Path
) -> None:
    """Reading files has been wrong before; the tool's own error is clearer."""

    _conda_tool(tmp_path, monkeypatch, torch=None)
    result = runner.invoke(
        app, ["proteinmpnn", "design", str(tiny_pdb), "2", "out", "--dry-run"]
    )
    assert result.exit_code == 0, result.output

    check = get_backends()["proteinmpnn"].check_environment(
        load_settings().tools["proteinmpnn"]
    )
    assert check.found
    assert any("no PyTorch was found" in warning for warning in check.warnings)
    assert any("whl/cu124" in remedy for remedy in check.remedies)


def test_a_cpu_only_torch_warns_but_does_not_block_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_pdb: Path
) -> None:
    """Slow is not broken: the researcher decides whether to run anyway."""

    checkout = _conda_tool(
        tmp_path, monkeypatch, torch="__version__ = '2.3.1'\ncuda = None\n"
    )
    script = checkout / "protein_mpnn_run.py"
    script.write_text("import sys\nprint('ran on cpu')\n")
    monkeypatch.setenv(
        "STRUCTBIO_USER_CONFIG", str(tmp_path / "user.yaml")
    )
    result = runner.invoke(
        app, ["proteinmpnn", "design", str(tiny_pdb), "2", "out", "--dry-run"]
    )
    assert result.exit_code == 0, result.output

    check = get_backends()["proteinmpnn"].check_environment(
        load_settings().tools["proteinmpnn"]
    )
    assert check.found
    assert any("CPU-only build" in warning for warning in check.warnings)


def test_doctor_marks_a_working_but_warned_installation_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _conda_tool(tmp_path, monkeypatch, torch="__version__ = '2.3.1'\ncuda = None\n")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "FOUND, WITH WARNINGS" in result.output
    assert "warning:" in result.output


def test_a_genuinely_unreachable_tool_points_at_env_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tiny_pdb: Path
) -> None:
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  proteinmpnn:\n"
        f"    path: {tmp_path / 'absent'}\n"
        "    executable: protein_mpnn_run.py\n"
        "    manager: conda\n"
        "    environment: mlfold\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["proteinmpnn", "design", str(tiny_pdb), "2", "out"])
    assert result.exit_code == 2
    assert "structbio env create proteinmpnn" in result.output


def test_env_create_dry_run_builds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _conda_tool(tmp_path, monkeypatch, torch=None)
    calls: list[object] = []
    monkeypatch.setattr("structbio.cli.subprocess.run", lambda *a, **k: calls.append(a))
    result = runner.invoke(app, ["env", "create", "proteinmpnn", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "conda create -y -n mlfold" in result.output
    assert "Dry run: nothing was built" in result.output
    assert calls == []


def test_env_create_refuses_when_no_combination_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "RFdiffusion"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "scripts" / "run_inference.py").touch()
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  rfdiffusion:\n"
        f"    path: {checkout}\n"
        "    executable: scripts/run_inference.py\n"
        "    manager: conda\n"
        "    environment: SE3nv\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.setattr("structbio.environment.gpu_capabilities", lambda: [(12, 0)])
    monkeypatch.setattr("structbio.environment.driver_cuda_version", lambda: (13, 0))

    result = runner.invoke(app, ["env", "create", "rfdiffusion"])
    assert result.exit_code == 1
    assert "Cannot build this environment" in result.output
    assert "What you can do instead" in result.output


def test_env_verify_needs_the_environment_to_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _conda_tool(tmp_path, monkeypatch, torch=None)
    monkeypatch.setattr("structbio.environment.conda_environments", dict)
    result = runner.invoke(app, ["env", "verify", "proteinmpnn"])
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_env_create_stops_rather_than_rebuilding_over_a_live_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebuilding on top of a live prefix is how a stale Python survives."""

    _conda_tool(tmp_path, monkeypatch, torch=None)
    monkeypatch.setattr("structbio.provision.environment_exists", lambda name: True)
    monkeypatch.setattr("structbio.provision.move_aside", lambda name: None)
    result = runner.invoke(app, ["env", "create", "proteinmpnn", "--force", "--yes"])
    assert result.exit_code == 2
    assert "could not be renamed" in result.output
    assert "Nothing was changed" in result.output


def test_env_create_keeps_the_previous_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--force must never destroy an environment somebody may depend on."""

    _conda_tool(tmp_path, monkeypatch, torch=None)
    seen: list[str] = []
    monkeypatch.setattr(
        "structbio.provision.environment_exists", lambda name: not seen
    )
    monkeypatch.setattr(
        "structbio.provision.move_aside",
        lambda name: seen.append(name) or f"{name}-before-1",
    )
    monkeypatch.setattr("structbio.provision.unusable_python", lambda name: None)
    monkeypatch.setattr("structbio.provision.environment_facts", lambda name: {})

    class _Process:
        returncode = 0
        stdout = iter(())

        def wait(self):
            return 0

    monkeypatch.setattr("structbio.cli.subprocess.Popen", lambda *a, **k: _Process())
    monkeypatch.setattr(
        "structbio.provision.verify",
        lambda tool, name, **k: __import__(
            "structbio.provision", fromlist=["x"]
        ).ProbeResult(ok=False, error="stopped here"),
    )
    result = runner.invoke(app, ["env", "create", "proteinmpnn", "--force", "--yes"])
    assert "kept as mlfold-before-1" in result.output
    assert "conda rename -n mlfold-before-1 mlfold" in result.output


def test_env_adopt_records_an_environment_that_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The right answer when a setup already works is to use it."""

    _conda_tool(tmp_path, monkeypatch, torch=None)
    monkeypatch.setattr("structbio.provision.environment_exists", lambda name: True)
    monkeypatch.setattr(
        "structbio.provision.environment_facts",
        lambda name: {"python": "3.10.21", "executable": "/envs/works/bin/python"},
    )
    monkeypatch.setattr(
        "structbio.provision.verify",
        lambda tool, name, **k: __import__("structbio.provision", fromlist=["x"]).ProbeResult(
            ok=True,
            values={
                "torch": "2.3.1+cu118",
                "torch_cuda": "11.8",
                "cuda_available": True,
                "device": "NVIDIA GeForce RTX 4090",
                "gpu_allocation": True,
                "numpy": True,
            },
        ),
    )
    result = runner.invoke(
        app, ["env", "adopt", "proteinmpnn", "--environment", "works"]
    )
    assert result.exit_code == 0, result.output
    assert "RTX 4090" in result.output
    written = yaml.safe_load((tmp_path / "user.yaml").read_text())
    assert written["tools"]["proteinmpnn"]["environment"] == "works"


def test_env_adopt_records_nothing_when_the_check_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _conda_tool(tmp_path, monkeypatch, torch=None)
    before = (tmp_path / "user.yaml").read_text()
    monkeypatch.setattr("structbio.provision.environment_exists", lambda name: True)
    monkeypatch.setattr("structbio.provision.environment_facts", lambda name: {})
    monkeypatch.setattr(
        "structbio.provision.verify",
        lambda tool, name, **k: __import__("structbio.provision", fromlist=["x"]).ProbeResult(
            ok=True, values={"torch_error": "No module named 'torch'"}
        ),
    )
    result = runner.invoke(
        app, ["env", "adopt", "proteinmpnn", "--environment", "broken"]
    )
    assert result.exit_code == 1
    assert "Not recorded" in result.output
    assert (tmp_path / "user.yaml").read_text() == before


def test_env_create_will_not_replace_an_environment_that_works(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even with --force: a working environment is the thing being protected."""

    _conda_tool(tmp_path, monkeypatch, torch=None)
    monkeypatch.setattr("structbio.provision.environment_exists", lambda name: True)
    monkeypatch.setattr(
        "structbio.provision.verify",
        lambda tool, name, **k: __import__(
            "structbio.provision", fromlist=["x"]
        ).ProbeResult(
            ok=True,
            values={
                "torch": "1.9.1",
                "torch_cuda": "11.1",
                "cuda_available": True,
                "device": "NVIDIA GeForce RTX 4090",
                "gpu_allocation": True,
                "numpy": True,
            },
        ),
    )
    moved: list[str] = []
    monkeypatch.setattr(
        "structbio.provision.move_aside", lambda name: moved.append(name) or "x"
    )
    result = runner.invoke(app, ["env", "create", "proteinmpnn", "--force", "--yes"])
    assert result.exit_code == 0, result.output
    assert "already works" in result.output
    assert "env adopt proteinmpnn" in result.output
    assert moved == []


def test_a_failed_run_shows_the_tools_own_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason belongs next to the failure, not only in a file."""

    checkout = tmp_path / "RFdiffusion" / "scripts"
    checkout.mkdir(parents=True)
    script = checkout / "run_inference.py"
    script.write_text(
        "import sys\n"
        "print('starting', flush=True)\n"
        "print('RuntimeError: CUDA out of memory', file=sys.stderr, flush=True)\n"
        "sys.exit(1)\n"
    )
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  rfdiffusion:\n"
        f"    path: {tmp_path / 'RFdiffusion'}\n"
        "    executable: scripts/run_inference.py\n"
        "    manager: none\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["rfdiffusion", "monomer", "40", "out", "--quiet"])
    assert result.exit_code == 1
    assert "RFdiffusion failed with exit code 1" in result.output
    assert "Last lines of stderr.log" in result.output
    assert "CUDA out of memory" in result.output

    # And again afterwards, from the recorded run.
    status = runner.invoke(app, ["status", "out"])
    assert status.exit_code == 0, status.output
    assert "CUDA out of memory" in status.output


def test_a_run_reports_which_model_files_it_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Detection output beside a final model looks like a broken prediction."""

    checkout = tmp_path / "RFdiffusion" / "scripts"
    checkout.mkdir(parents=True)
    (checkout / "run_inference.py").write_text(
        "import sys, pathlib\n"
        "prefix = [a.split('=', 1)[1] for a in sys.argv[1:]\n"
        "          if a.startswith('inference.output_prefix=')][0]\n"
        "atom = 'ATOM  {:5d}  CA  ALA {}{:4d}      11.104  13.207   9.180  1.00 20.00           C'\n"
        "pathlib.Path(prefix + '_0.pdb').write_text(\n"
        "    '\\n'.join(atom.format(i, 'A', i) for i in range(1, 6)) + '\\n')\n"
    )
    user_config = tmp_path / "user.yaml"
    user_config.write_text(
        "tools:\n"
        "  rfdiffusion:\n"
        f"    path: {tmp_path / 'RFdiffusion'}\n"
        "    executable: scripts/run_inference.py\n"
        "    manager: none\n"
    )
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(user_config))
    monkeypatch.setenv("STRUCTBIO_LAB_CONFIG", str(tmp_path / "absent.yaml"))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["rfdiffusion", "monomer", "5", "out", "--quiet"])
    assert result.exit_code == 0, result.output
    assert "Model files (1)" in result.output
    assert "out_0.pdb" in result.output
    assert "1 chains (A:5)" in result.output


def test_a_quick_run_finds_an_unconfigured_tool_by_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No configuration, no setup: the first command still has to work."""

    from structbio import autoconfig, discovery

    software = tmp_path / "software"
    (software / "RFdiffusion" / "scripts").mkdir(parents=True)
    (software / "RFdiffusion" / "scripts" / "run_inference.py").touch()
    monkeypatch.setattr(discovery, "conda_environments", dict)
    scan = discovery.discover
    monkeypatch.setattr(
        discovery,
        "discover",
        lambda sigs=None, **_: scan(sigs or discovery.SIGNATURES, roots=(str(software),)),
    )
    monkeypatch.delenv(autoconfig.DISABLE_VARIABLE, raising=False)
    config_path = tmp_path / "config.yaml"
    monkeypatch.setenv("STRUCTBIO_USER_CONFIG", str(config_path))
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["rfdiffusion", "monomer", "80", "designs", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "Found rfdiffusion at" in result.output
    assert str(software / "RFdiffusion") in config_path.read_text()
