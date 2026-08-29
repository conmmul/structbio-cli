"""The command tree, and the option help shared across it.

Kept apart from the commands themselves so that every command module can
import the group it registers on without importing the others.
"""

from __future__ import annotations

import typer


app = typer.Typer(
    name="structbio",
    help=(
        "Short commands for structural-biology software on a workstation.\n\n"
        "First time here?  structbio setup\n\n"
        "Quick form:  structbio TOOL RUNTYPE ... OUTPUT_FOLDER\n"
        "YAML form:   structbio TOOL run CONFIG.yaml\n\n"
        "For example: rfdiffusion monomer 150 my_designs -n 10"
    ),
    no_args_is_help=True,
)
rfdiffusion_app = typer.Typer(
    help=(
        "RFdiffusion backbone design.\n\n"
        "  rfdiffusion monomer LENGTH OUTPUT\n"
        "  rfdiffusion symmetry GROUP TOTAL_LENGTH OUTPUT\n"
        "  rfdiffusion binder TARGET.pdb LENGTH OUTPUT\n"
        "  rfdiffusion partial INPUT.pdb STEPS OUTPUT"
    ),
    no_args_is_help=True,
)
proteinmpnn_app = typer.Typer(
    help=(
        "ProteinMPNN sequence design.\n\n"
        "  proteinmpnn design INPUT.pdb NUM_SEQUENCES OUTPUT"
    ),
    no_args_is_help=True,
)
colabfold_app = typer.Typer(
    help=(
        "ColabFold structure prediction.\n\n"
        "  colabfold predict SEQUENCES OUTPUT"
    ),
    no_args_is_help=True,
)
cryozeta_app = typer.Typer(
    help=(
        "CryoZeta cryo-EM structure modelling.\n\n"
        "  cryozeta predict MAP.mrc CHAINS.fasta OUTPUT --resolution R --contour C\n"
        "  cryozeta predict-json TARGETS.json OUTPUT"
    ),
    no_args_is_help=True,
)
app.add_typer(rfdiffusion_app, name="rfdiffusion")
app.add_typer(proteinmpnn_app, name="proteinmpnn")
app.add_typer(colabfold_app, name="colabfold")
app.add_typer(cryozeta_app, name="cryozeta")


OUTPUT_HELP = "Output folder; its name is also used as the output file prefix"
GPU_HELP = "GPU to use: an id such as 0, several as 0,1, or 'auto' for the idlest"
QUIET_HELP = "Do not echo the tool's output; it is still written to the log"
SET_HELP = "Override any configuration value: dotted.key=YAML_VALUE"
