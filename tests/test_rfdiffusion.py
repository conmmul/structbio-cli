from pathlib import Path

from structbio.config import ToolInstallation
from structbio.tools.base import BackendContext
from structbio.tools.rfdiffusion import RFDiffusionBackend


def _context(tmp_path: Path) -> BackendContext:
    return BackendContext(
        source=tmp_path / "config.yaml",
        installation=ToolInstallation(
            path=Path("/opt/RFdiffusion"),
            environment="SE3nv",
            manager="conda",
            executable="scripts/run_inference.py",
        ),
        experiment_dir=tmp_path / "experiment",
        output_dir=tmp_path / "experiment/outputs",
        inputs_dir=tmp_path / "experiment/inputs",
    )


def test_symmetry_command_generation(tmp_path: Path) -> None:
    backend = RFDiffusionBackend()
    config = backend.parse_config(
        {
            "tool": "rfdiffusion",
            "experiment": {"name": "tetra600"},
            "design": {
                "mode": "symmetry",
                "symmetry": "tetrahedral",
                "length": 600,
                "num_designs": 20,
            },
            "potentials": {
                "olig_contacts": {"weight_intra": 1.0, "weight_inter": 0.05}
            },
            "diffusion": {"guide_scale": 0.5, "guide_decay": "quadratic"},
        },
        tmp_path / "config.yaml",
    )
    report = backend.validate(config)
    assert report.ok, report.errors
    command = backend.build_command(config, _context(tmp_path)).steps[0].argv
    assert command[:6] == (
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "SE3nv",
        "python",
    )
    assert "--config-name" in command
    assert "inference.symmetry=tetrahedral" in command
    assert "contigmap.contigs=[600]" in command
    assert "inference.num_designs=20" in command
    assert any(item.startswith("potentials.guiding_potentials=") for item in command)


def test_motif_references_are_validated(tmp_path: Path, tiny_pdb: Path) -> None:
    backend = RFDiffusionBackend()
    config = backend.parse_config(
        {
            "tool": "rfdiffusion",
            "experiment": {"name": "motif"},
            "input": {"pdb": str(tiny_pdb)},
            "design": {"mode": "motif", "contigs": ["5-10/Z1-2/5-10"]},
        },
        tmp_path / "config.yaml",
    )
    assert not backend.validate(config).ok
