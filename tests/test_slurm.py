from pathlib import Path

import pytest

from structbio.config import ClusterProfile, ResourceConfig
from structbio.slurm import generate_slurm_script
from structbio.tools.base import CommandPlan, CommandStep


def test_slurm_script_generation(tmp_path: Path) -> None:
    plan = CommandPlan(
        [CommandStep(("python", "tool.py", "--output", str(tmp_path / "outputs")))],
        output_dir=tmp_path / "outputs",
    )
    script = generate_slurm_script(
        plan,
        experiment_dir=tmp_path,
        job_name="design",
        resources=ResourceConfig(gpus=4, cpus=32, memory="64G", time="04:00:00"),
        profile=ClusterProfile(partition="gpu", modules=["cuda"]),
    )
    assert "#SBATCH --gpus=4" in script
    assert "#SBATCH --cpus-per-task=32" in script
    assert "#SBATCH --partition=gpu" in script
    assert "module load cuda" in script
    assert "CUDA_VISIBLE_DEVICES" not in script


def test_invalid_slurm_array_is_rejected(tmp_path: Path) -> None:
    plan = CommandPlan([CommandStep(("true",))], output_dir=tmp_path)
    with pytest.raises(ValueError, match="array"):
        generate_slurm_script(
            plan,
            experiment_dir=tmp_path,
            job_name="bad",
            resources=ResourceConfig(array="$(danger)"),
        )


def test_slurm_array_selects_one_independent_step(tmp_path: Path) -> None:
    plan = CommandPlan(
        [
            CommandStep(("python", "design.py", "one.pdb"), name="one"),
            CommandStep(("python", "design.py", "two.pdb"), name="two"),
        ],
        output_dir=tmp_path,
    )
    script = generate_slurm_script(
        plan,
        experiment_dir=tmp_path,
        job_name="batch",
        resources=ResourceConfig(array="0-1"),
    )
    assert "#SBATCH --array=0-1" in script
    assert 'case "${SLURM_ARRAY_TASK_ID}" in' in script
    assert "0) python design.py one.pdb ;;" in script
    assert "1) python design.py two.pdb ;;" in script


def test_array_rejects_single_command_to_prevent_duplicate_work(tmp_path: Path) -> None:
    plan = CommandPlan([CommandStep(("true",))], output_dir=tmp_path)
    with pytest.raises(ValueError, match="independent steps"):
        generate_slurm_script(
            plan,
            experiment_dir=tmp_path,
            job_name="duplicate",
            resources=ResourceConfig(array="0-9"),
        )
