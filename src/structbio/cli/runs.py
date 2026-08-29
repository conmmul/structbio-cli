"""Quick commands: positional arguments, plain output folders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError

from structbio import quick
from structbio.cli.app import (
    GPU_HELP,
    OUTPUT_HELP,
    QUIET_HELP,
    SET_HELP,
    colabfold_app,
    cryozeta_app,
    proteinmpnn_app,
    rfdiffusion_app,
)
from structbio.cli.support import (
    _abort,
    _backend_for,
    _context,
    _cryozeta_gpu,
    _expected_chains,
    _report_outputs,
    _require_installed,
    _show_failure,
    _show_report,
    _with_gpu,
)
from structbio.config import load_command_line_config, parse_cli_overrides
from structbio.execution import execute_plan, plan_input_paths
from structbio.experiment import direct_paths, prepare_output_dir, safe_name, write_records


def _quick_run(
    tool: str,
    fragment_builder: Any,
    output: Path,
    *,
    gpu: str | None,
    dry_run: bool,
    set_values: list[str],
    quiet: bool = False,
    **arguments: Any,
) -> None:
    """Validate and run a positional command against a plain output folder."""

    output_dir = output.expanduser()
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    try:
        name = safe_name(output_dir.name)
    except ValueError:
        _abort(f"Output folder name must contain letters or numbers: {output_dir.name!r}")
    try:
        fragment = fragment_builder(name=name, **arguments)
        loaded = load_command_line_config(
            fragment, overrides=parse_cli_overrides(set_values)
        )
        loaded, backend, config = _backend_for(loaded, tool)
        report = backend.validate(config)
    except (ValueError, ValidationError) as exc:
        _abort(str(exc))
    if not report.ok:
        _show_report(report)
        raise typer.Exit(2)

    if dry_run:
        plan = _with_gpu(
            backend.build_command(config, _context(loaded, backend, direct_paths(output_dir))),
            gpu,
        )
        _show_report(report)
        typer.echo(f"\nCommand:\n{plan.render()}")
        typer.echo(f"\nOutput folder: {output_dir}")
        typer.echo("Dry run: nothing was created or executed.")
        return

    installation = _require_installed(loaded, backend)
    try:
        paths = prepare_output_dir(output_dir)
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    plan = _with_gpu(backend.build_command(config, _context(loaded, backend, paths)), gpu)
    backend.materialize_artifacts(plan)
    write_records(
        paths,
        config=loaded.data,
        command=plan.render(),
        tool_name=backend.name,
        tool_path=installation.path,
        input_paths=plan_input_paths(plan),
        status="prepared",
    )
    typer.echo(f"{backend.display_name}: {plan.render()}")
    typer.echo(f"Output folder: {paths.outputs}")
    return_code = execute_plan(plan, paths, stream=not quiet)
    if return_code:
        _show_failure(backend.display_name, return_code, paths)
        raise typer.Exit(return_code)
    _report_outputs(paths.outputs, _expected_chains(config))
    typer.echo(f"\nDone. Results are in {paths.outputs}")


@rfdiffusion_app.command("monomer")
def rfdiffusion_monomer(
    length: int = typer.Argument(..., min=1, help="Residues per design"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    num: int = typer.Option(1, "-n", "--num", min=1, help="Number of designs"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Design new monomer backbones from nothing but a length.

    Example: rfdiffusion monomer 150 my_monomers -n 10
    """

    _quick_run(
        "rfdiffusion",
        quick.rfdiffusion_monomer,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        length=length,
        num_designs=num,
    )


@rfdiffusion_app.command("symmetry")
def rfdiffusion_symmetry(
    group: str = typer.Argument(..., help="Symmetry group: c2..cN, d2..dN, or tetrahedral"),
    length: int = typer.Argument(..., min=1, help="Total residues across every subunit"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    num: int = typer.Option(1, "-n", "--num", min=1, help="Number of designs"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Design symmetric oligomers.

    The length is the total across all subunits and must divide by the subunit
    count. Example: rfdiffusion symmetry c4 400 my_tetramers
    """

    _quick_run(
        "rfdiffusion",
        quick.rfdiffusion_symmetry,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        symmetry=group,
        length=length,
        num_designs=num,
    )


@rfdiffusion_app.command("binder")
def rfdiffusion_binder(
    target: Path = typer.Argument(..., exists=True, dir_okay=False, help="Target structure"),
    length: int = typer.Argument(..., min=1, help="Residues in the designed binder"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    chain: str | None = typer.Option(
        None, "--chain", help="Target chain; required when the file has several"
    ),
    hotspots: str | None = typer.Option(
        None, "--hotspots", help="Target residues to engage, such as B30,B33,B34"
    ),
    num: int = typer.Option(1, "-n", "--num", min=1, help="Number of designs"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Design binders against a target structure.

    The target contig is read from the residue numbering in the file itself.
    Example: rfdiffusion binder target.pdb 100 my_binders --hotspots B30,B33
    """

    _quick_run(
        "rfdiffusion",
        quick.rfdiffusion_binder,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        target=target,
        length=length,
        num_designs=num,
        chain=chain,
        hotspots=hotspots,
    )


@rfdiffusion_app.command("partial")
def rfdiffusion_partial(
    pdb: Path = typer.Argument(..., exists=True, dir_okay=False, help="Structure to diversify"),
    steps: int = typer.Argument(..., min=1, help="Partial diffusion timesteps"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    num: int = typer.Option(1, "-n", "--num", min=1, help="Number of designs"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Diversify an existing structure by partial diffusion.

    Every chain is kept at its own length. Fewer steps stay closer to the input.
    Example: rfdiffusion partial start.pdb 10 my_variants -n 20
    """

    _quick_run(
        "rfdiffusion",
        quick.rfdiffusion_partial,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        pdb=pdb,
        steps=steps,
        num_designs=num,
    )


@proteinmpnn_app.command("design")
def proteinmpnn_design(
    input_path: Path = typer.Argument(..., exists=True, help="PDB file, or a folder of PDBs"),
    num_sequences: int = typer.Argument(..., min=1, help="Sequences per structure"),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    chains: str | None = typer.Option(None, "--chains", help="Chains to design, such as A,B"),
    designable: str | None = typer.Option(
        None, "--designable", help="Only these positions may change, such as A:697-749"
    ),
    fixed: str | None = typer.Option(
        None, "--fixed", help="Positions that must not change, such as A:700,A:705"
    ),
    temperature: str | None = typer.Option(
        None, "--temp", help="Sampling temperature(s), default 0.1"
    ),
    seed: int = typer.Option(0, "--seed", min=0, help="Random seed"),
    soluble: bool = typer.Option(False, "--soluble", help="Use the soluble-protein model"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Design sequences for a structure, or for every PDB in a folder.

    Without --designable every residue of the selected chains may change.
    Example: proteinmpnn design 7kdp.pdb 8 my_sequences --designable A:697-749
    """

    _quick_run(
        "proteinmpnn",
        quick.proteinmpnn_design,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        input_path=input_path,
        num_sequences=num_sequences,
        chains=chains,
        designable=designable,
        fixed=fixed,
        temperature=temperature,
        seed=seed,
        soluble=soluble,
    )


@colabfold_app.command("predict")
def colabfold_predict(
    sequences: Path = typer.Argument(
        ..., exists=True, help="FASTA file, a folder of them, or a ProteinMPNN output folder"
    ),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    num_models: int = typer.Option(
        5, "-n", "--num-models", min=1, max=5, help="Models per sequence"
    ),
    msa_mode: str = typer.Option(
        "mmseqs2_uniref_env",
        "--msa-mode",
        help="mmseqs2_uniref_env, mmseqs2_uniref_env_envpair, mmseqs2_uniref, or single_sequence",
    ),
    templates: bool = typer.Option(False, "--templates", help="Use PDB templates"),
    relax: int = typer.Option(
        0, "--relax", min=0, max=5, help="Amber-relax this many top-ranked models"
    ),
    recycle: int | None = typer.Option(None, "--recycle", min=0, help="Recycle iterations"),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Predict structures for designed or natural sequences.

    Point it at a ProteinMPNN output folder and it finds the designed sequences
    itself. By default MSAs come from the public ColabFold server, which means
    the sequences leave this machine; --msa-mode single_sequence keeps them local.

    Example: colabfold predict my_sequences my_folds -n 5 --relax 1
    """

    _quick_run(
        "colabfold",
        quick.colabfold_predict,
        output,
        gpu=gpu,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        sequences=sequences,
        num_models=num_models,
        msa_mode=msa_mode,
        templates=templates,
        relax=relax,
        num_recycle=recycle,
    )


@cryozeta_app.command("predict")
def cryozeta_predict(
    density_map: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Cryo-EM density map (.map, .mrc, .map.gz)"
    ),
    sequences: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="FASTA of every chain in the complex"
    ),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    resolution: float = typer.Option(
        ..., "--resolution", min=0.1, max=30.0, help="Map resolution in angstroms"
    ),
    contour: float = typer.Option(
        ..., "--contour", help="Recommended contour level for the map"
    ),
    mode: str = typer.Option(
        "combined", "--mode", help="combined, cryozeta, or cryozeta-interpolate"
    ),
    large: bool = typer.Option(
        False, "--large", help="Use the large-complex pipeline (above ~2800 residues)"
    ),
    registration: str = typer.Option(
        "auto", "--registration", help="Large-complex registration: auto, teaser, svd, vesper"
    ),
    dna: str | None = typer.Option(None, "--dna", help="FASTA record names that are DNA"),
    rna: str | None = typer.Option(None, "--rna", help="FASTA record names that are RNA"),
    protein: str | None = typer.Option(
        None, "--protein", help="FASTA record names that are protein, when ambiguous"
    ),
    msa_dir: Path | None = typer.Option(
        None, "--msa-dir", help="Directory of precomputed .a3m files"
    ),
    pairing_db: str | None = typer.Option(
        None, "--pairing-db", help="MSA pairing database, such as uniref100"
    ),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Model a structure into a cryo-EM map from the map and its sequences.

    The native CryoZeta target JSON is written for you into
    OUTPUT/.structbio/inputs/targets.json. Identical sequences are counted as
    copies of one chain. Ligands, ions, and modifications need a hand-written
    JSON and `cryozeta predict-json`.

    Example: cryozeta predict emd_44046.map.gz chains.fasta my_model
             --resolution 2.99 --contour 0.3
    """

    _quick_run(
        "cryozeta",
        quick.cryozeta_predict,
        output,
        gpu=None,  # CryoZeta takes GPU ids on its own command line
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        density_map=density_map,
        sequences=sequences,
        resolution=resolution,
        contour=contour,
        mode=mode,
        large=large,
        registration=registration,
        dna=dna,
        rna=rna,
        protein=protein,
        msa_dir=msa_dir,
        pairing_db=pairing_db,
        gpu_ids=_cryozeta_gpu(gpu),
    )


@cryozeta_app.command("predict-json")
def cryozeta_predict_json(
    targets: Path = typer.Argument(
        ..., exists=True, dir_okay=False, help="Native CryoZeta target JSON"
    ),
    output: Path = typer.Argument(..., help=OUTPUT_HELP),
    mode: str = typer.Option(
        "combined", "--mode", help="combined, cryozeta, or cryozeta-interpolate"
    ),
    large: bool = typer.Option(
        False, "--large", help="Use the large-complex pipeline (above ~2800 residues)"
    ),
    registration: str = typer.Option(
        "auto", "--registration", help="Large-complex registration: auto, teaser, svd, vesper"
    ),
    gpu: str | None = typer.Option(None, "--gpu", help=GPU_HELP),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show the command only"),
    quiet: bool = typer.Option(False, "--quiet", help=QUIET_HELP),
    set_value: list[str] = typer.Option([], "--set", help=SET_HELP),
) -> None:
    """Run a CryoZeta target list you wrote yourself.

    Use this for anything the short form cannot describe: several targets in one
    run, ligands and ions, chain modifications, or glycans.

    Example: cryozeta predict-json targets.json my_models --gpu 0
    """

    _quick_run(
        "cryozeta",
        quick.cryozeta_predict_json,
        output,
        gpu=None,
        dry_run=dry_run,
        quiet=quiet,
        set_values=set_value,
        input_json=targets,
        mode=mode,
        large=large,
        registration=registration,
        gpu_ids=_cryozeta_gpu(gpu),
    )

