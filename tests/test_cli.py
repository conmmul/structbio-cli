from pathlib import Path

from typer.testing import CliRunner

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
