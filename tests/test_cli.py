import json
from pathlib import Path

import pytest
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
    monkeypatch.setattr("structbio.cli.select_idle_gpu", lambda: 2)
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
    monkeypatch.setattr("structbio.cli.select_idle_gpu", lambda: None)
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
