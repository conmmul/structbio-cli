# Troubleshooting

Start with `structbio doctor`, then inspect the generated command using
`structbio TOOL command CONFIG.yaml`.

## Installation diagnostics

### NOT CONFIGURED

Add the tool path, executable, manager, and environment to the lab or user
configuration. Confirm that `path/executable` names the actual installed script,
not an example command from another checkout version.

### CONFIGURED, UNAVAILABLE

The configured script or environment manager is missing on this host. Check:

```bash
structbio doctor
ls -l /configured/tool/path/relative/executable
conda env list
pixi --version
```

Only run commands that apply to the configured manager. A tool can be available
on a compute node even if it is unavailable on a laptop; prepare the run on the
host that has access to the shared installation.

## Configuration errors

### A relative input path points to the wrong file

Experiment input paths are resolved relative to the experiment YAML. Keep input
files near a project-specific YAML or use an absolute shared-filesystem path.
The experiments output root, by contrast, defaults to `./experiments` relative
to the directory where `structbio` is invoked.

### A temporary change should not be saved in YAML

Use a typed command-line override:

```bash
structbio rfdiffusion command design.yaml --set design.num_designs=2
structbio proteinmpnn run mpnn.yaml --dry-run --set resources.gpus=1
```

The value after `=` is parsed as YAML, so numbers and booleans retain their
types. Repeat `--set` for multiple keys.

## RFdiffusion validation errors

- **PDB missing:** motif, binder, partial, and inpainting modes require
  `input.pdb`.
- **Missing chain/residue:** inspect the original PDB. `structbio` will not
  renumber residues or guess a chain.
- **Malformed contig:** keep each contig as a single YAML scalar and preserve
  RFdiffusion's `/0 ` chain-break space.
- **Unsupported symmetry:** use `cN`, `dN`, or `tetrahedral` with a valid order.
- **Partial diffusion failure:** check both `partial_t <= timesteps` and the
  upstream requirement that the contig length match the partially diffused
  input.
- **Hydra/runtime failure after validation:** compare `command.txt` with the
  installed RFdiffusion documentation and verify checkpoint compatibility.

## ProteinMPNN mask errors

- Run `inspect-mask` before every new selection.
- Use original PDB numbering in `CHAIN:START-END` format.
- A requested inclusive range cannot contain missing residue numbers.
- Designable and fixed selections cannot overlap.
- A directory batch requires the requested chains/ranges to exist in every PDB.
- `num_sequences` must be divisible by `batch_size`.

If the reported mask is not what was intended, stop and edit the YAML. Do not
invert the selection manually or renumber the raw PDB to satisfy the wrapper.

## CryoZeta errors

- Validate the native input JSON against the installed CryoZeta version.
- Prefer absolute map and MSA paths that are visible on compute nodes.
- Confirm Pixi assets, checkpoints, CUDA toolchain, and compiler cache locations.
- The standard wrapped pipeline is single-GPU; requesting extra GPUs does not
  parallelize it.
- Large-complex mode is not currently wrapped.

## Experiment and SLURM behavior

### An experiment name already exists

This is normal. A new numeric suffix is selected automatically; existing records
are never overwritten. Use the exact printed directory name with
`structbio status`.

### `submit` did not send a job

This is the safe default. `submit CONFIG.yaml` writes `job.slurm` and stops. Use
`submit CONFIG.yaml --execute` to create and submit a new experiment.

### SLURM is unavailable

Generate and inspect scripts on a workstation or login node. Actual submission
requires a host with `sbatch`. Live status requires `squeue`.

### A completed or failed job is absent from `squeue`

`squeue` normally shows active jobs only. Inspect the experiment logs and use
the cluster's accounting command according to local policy.
