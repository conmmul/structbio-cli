"""Guided installation of the wrapped scientific software.

structbio clones a project and then tells you the remaining steps in that
project's own words. It deliberately stops there. Environment creation and
model weights differ per machine, change with each upstream release, and in
some cases carry a licence that only the researcher can accept, so
reimplementing them here would go stale and take a decision that is not ours.

Every recipe below was read from the upstream README on the date recorded in
`verified_on`.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstallRecipe:
    tool: str
    display_name: str
    repository: str
    directory_name: str
    verified_on: str
    licence: str
    weights: str
    steps: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def target(self, into: Path) -> Path:
        return into.expanduser() / self.directory_name

    def rendered_steps(self, directory: Path) -> list[str]:
        return [step.format(directory=directory) for step in self.steps]


RFDIFFUSION_WEIGHTS = (
    "6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
    "e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
    "60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt",
    "74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt",
    "76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt",
    "5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt",
    "12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt",
)


RECIPES: dict[str, InstallRecipe] = {
    "rfdiffusion": InstallRecipe(
        tool="rfdiffusion",
        display_name="RFdiffusion",
        repository="https://github.com/RosettaCommons/RFdiffusion.git",
        directory_name="RFdiffusion",
        verified_on="2026-08-28",
        licence="BSD; free for both non-profit and for-profit use.",
        weights="Seven checkpoints from files.ipd.uw.edu, downloaded by the steps below.",
        steps=(
            "conda env create -f {directory}/env/SE3nv.yml",
            "conda activate SE3nv",
            "cd {directory}/env/SE3Transformer",
            "pip install --no-cache-dir -r requirements.txt",
            "python setup.py install",
            "cd {directory}",
            "pip install -e .",
            "mkdir -p {directory}/models && cd {directory}/models",
            *(
                f"wget http://files.ipd.uw.edu/pub/RFdiffusion/{item}"
                for item in RFDIFFUSION_WEIGHTS
            ),
        ),
        notes=(
            "The upstream README expects the whole setup to take under 30 minutes.",
            "Re-run 'structbio setup' afterwards to record the path.",
        ),
    ),
    "proteinmpnn": InstallRecipe(
        tool="proteinmpnn",
        display_name="ProteinMPNN",
        repository="https://github.com/dauparas/ProteinMPNN.git",
        directory_name="ProteinMPNN",
        verified_on="2026-08-28",
        licence="MIT; see the repository LICENSE file.",
        weights=(
            "Included in the repository: vanilla_model_weights/, "
            "soluble_model_weights/, and ca_model_weights/. Nothing else to download."
        ),
        steps=(
            "conda create --name mlfold",
            "conda activate mlfold",
            "# Take the exact PyTorch line for this machine from https://pytorch.org,",
            "# for example:",
            "conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch",
        ),
        notes=(
            "ProteinMPNN needs only Python, PyTorch and NumPy, and runs on the CPU "
            "when no GPU is present.",
        ),
    ),
    "colabfold": InstallRecipe(
        tool="colabfold",
        display_name="ColabFold",
        repository="https://github.com/YoshitakaMo/localcolabfold.git",
        directory_name="localcolabfold",
        verified_on="2026-08-28",
        licence="MIT for the installer; AlphaFold parameters carry their own terms.",
        weights="AlphaFold parameters are fetched by ColabFold itself on the first run.",
        steps=(
            "curl -fsSL https://pixi.sh/install.sh | sh",
            "cd {directory}",
            "pixi install && pixi run setup",
        ),
        notes=(
            "This installs colabfold_batch into "
            "{directory}/.pixi/envs/default/bin, which 'structbio setup' "
            "will find.",
            "localcolabfold targets Linux and macOS; the CUDA build needs an "
            "NVIDIA GPU.",
        ),
    ),
    "cryozeta": InstallRecipe(
        tool="cryozeta",
        display_name="CryoZeta",
        repository="https://github.com/kiharalab/CryoZeta.git",
        directory_name="CryoZeta",
        verified_on="2026-08-28",
        licence=(
            "Source code GPL-3.0. The model weights are free for academic and "
            "non-commercial research use ONLY; commercial use needs the authors' "
            "permission. Read WEIGHT_LICENSE.md in the checkout before running it."
        ),
        weights="Downloaded from Hugging Face by 'pixi run setup'.",
        steps=(
            "curl -fsSL https://pixi.sh/install.sh | bash",
            "cd {directory}",
            "pixi run setup",
        ),
        notes=(
            "'pixi run setup' installs dependencies, detects the GPU and picks the "
            "matching CUDA version, downloads the weights and examples, and builds "
            "TEASER++.",
        ),
    ),
}


def recipe_for(tool: str) -> InstallRecipe:
    try:
        return RECIPES[tool]
    except KeyError as exc:
        known = ", ".join(sorted(RECIPES))
        raise ValueError(f"No install recipe for {tool!r}; known tools: {known}") from exc


def clone(recipe: InstallRecipe, into: Path) -> Path:
    """Clone one project, refusing to touch a directory that already exists."""

    if shutil.which("git") is None:
        raise RuntimeError("git is not installed, so the repository cannot be cloned")
    into = into.expanduser()
    target = recipe.target(into)
    if target.exists():
        raise FileExistsError(
            f"{target} already exists; remove it, or point --into somewhere else"
        )
    into.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", recipe.repository, str(target)], check=False
    )
    if result.returncode:
        raise RuntimeError(f"git clone failed with exit code {result.returncode}")
    return target
