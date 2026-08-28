import json
from pathlib import Path

from structbio.config import ToolInstallation
from structbio.tools.base import BackendContext
from structbio.tools.cryozeta import CryoZetaBackend


def test_cryozeta_uses_verified_official_script_interface(tmp_path: Path) -> None:
    density_map = tmp_path / "target.map"
    density_map.write_bytes(b"map")
    input_json = tmp_path / "input.json"
    input_json.write_text(
        json.dumps(
            [
                {
                    "name": "target",
                    "modelSeeds": [],
                    "map_path": "target.map",
                    "resolution": 3.0,
                    "contour_level": 0.2,
                    "sequences": [{"dnaSequence": {"sequence": "ACGT", "count": 1}}],
                }
            ]
        )
    )
    backend = CryoZetaBackend()
    config = backend.parse_config(
        {
            "tool": "cryozeta",
            "experiment": {"name": "target"},
            "input": {"json": str(input_json)},
            "mode": "combined",
            "pixi_environment": "cu11",
        },
        tmp_path / "dataset.yaml",
    )
    assert backend.validate(config).ok
    context = BackendContext(
        source=tmp_path / "dataset.yaml",
        installation=ToolInstallation(
            path=Path("/opt/CryoZeta"),
            executable="inference_demo.sh",
            manager="pixi",
        ),
        experiment_dir=tmp_path / "experiment",
        output_dir=tmp_path / "experiment/outputs",
        inputs_dir=tmp_path / "experiment/inputs",
    )
    argv = backend.build_command(config, context).steps[0].argv
    assert argv[:2] == ("bash", "/opt/CryoZeta/inference_demo.sh")
    assert "--input-json" in argv
    assert "--output-dir" in argv
    assert "--mode" in argv
    assert "--env" in argv
    assert "--overwrite" not in argv
