import subprocess
from pathlib import Path

import pytest

from structbio import install
from structbio.tools import get_backends


def _local_repository(tmp_path: Path) -> Path:
    """A real git repository on disk, so cloning is tested without a network."""

    source = tmp_path / "upstream"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "run_inference.py").write_text("print('hello')\n")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "initial"],
    ):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    return source


def _recipe(source: Path) -> install.InstallRecipe:
    return install.InstallRecipe(
        tool="rfdiffusion",
        display_name="RFdiffusion",
        repository=str(source),
        directory_name="RFdiffusion",
        verified_on="2026-08-28",
        licence="BSD",
        weights="none for this test",
        steps=("conda env create -f {directory}/env/SE3nv.yml",),
    )


def test_every_backend_has_an_install_recipe() -> None:
    assert set(install.RECIPES) == set(get_backends())


def test_recipes_name_their_licence_and_verification_date() -> None:
    for recipe in install.RECIPES.values():
        assert recipe.licence and recipe.weights
        assert recipe.verified_on
        assert recipe.repository.startswith("https://github.com/")


def test_the_cryozeta_recipe_states_its_non_commercial_weight_licence() -> None:
    licence = install.RECIPES["cryozeta"].licence
    assert "non-commercial" in licence
    assert "WEIGHT_LICENSE" in licence


def test_steps_are_rendered_against_the_target_directory() -> None:
    recipe = install.RECIPES["rfdiffusion"]
    steps = recipe.rendered_steps(Path("/opt/RFdiffusion"))
    assert "conda env create -f /opt/RFdiffusion/env/SE3nv.yml" in steps
    assert not any("{directory}" in step for step in steps)


def test_clone_creates_the_checkout(tmp_path: Path) -> None:
    recipe = _recipe(_local_repository(tmp_path))
    target = install.clone(recipe, tmp_path / "software")
    assert target == tmp_path / "software" / "RFdiffusion"
    assert (target / "scripts" / "run_inference.py").is_file()


def test_clone_refuses_an_existing_directory(tmp_path: Path) -> None:
    recipe = _recipe(_local_repository(tmp_path))
    (tmp_path / "software" / "RFdiffusion").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        install.clone(recipe, tmp_path / "software")


def test_unknown_tools_are_reported_with_the_known_ones() -> None:
    with pytest.raises(ValueError, match="known tools"):
        install.recipe_for("alphafold")
