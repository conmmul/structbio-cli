# Troubleshooting

Start with `structbio doctor`, then re-run the command with `--dry-run` to see
exactly what would be executed.

## Installation diagnostics

### NOT CONFIGURED

Add the tool path, executable, manager, and environment to the user
configuration; `structbio config` prints which file is in use, and
`structbio setup` creates it from a template. Confirm that `path/executable`
names the actual installed script, not an example command from another checkout
version.

### CONFIGURED, UNAVAILABLE

The configured script or environment manager is missing on this host. Check:

```bash
structbio doctor
ls -l /configured/tool/path/relative/executable
conda env list
pixi --version
```

Only run commands that apply to the configured manager. A tool that lives on a
shared filesystem is only usable from a machine that mounts it.

## Configuration errors

### A relative input path points to the wrong file

Paths typed on the command line are resolved relative to the current directory,
which is normally what you want. Paths inside a YAML file are resolved relative
to that file, so keep input files near a project-specific YAML or use absolute
paths.

### A temporary change should not be saved in YAML

Use a typed command-line override:

```bash
rfdiffusion monomer 150 my_monomers --set diffusion.timesteps=50
structbio rfdiffusion command design.yaml --set design.num_designs=2
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
- Prefer absolute map and MSA paths so they do not depend on the current
  directory.
- Confirm Pixi assets, checkpoints, CUDA toolchain, and compiler cache locations.
- The standard wrapped pipeline is single-GPU; requesting extra GPUs does not
  parallelize it.
- Large-complex mode is not currently wrapped.

## Output folders and experiments

### Refusing to write into an existing non-empty folder

A quick command writes results into the folder you name and will not write over
anything already there. Choose a new name, or delete the old folder yourself
after checking what is in it.

### Where a run recorded what it did

`OUTPUT/.structbio/` for a quick command, and the experiment directory for a
YAML run. Either way, `structbio status FOLDER` prints the summary and
`command.txt` holds the exact command.

### An experiment name already exists

This is normal. A new numeric suffix is selected automatically; existing records
are never overwritten. Use the exact printed directory name with
`structbio status`.

### `submit` did not send a job

This is the safe default. `submit CONFIG.yaml` writes `job.slurm` and stops. Use
`submit CONFIG.yaml --execute` to create and submit a new experiment.

### SLURM is unavailable

Nothing on a workstation needs it. `submit` exists only for shared clusters:
generating a script needs nothing, submitting it needs `sbatch`, and live status
needs `squeue`. See [the optional cluster guide](cluster.md).

### A completed or failed job is absent from `squeue`

`squeue` normally shows active jobs only. Inspect the experiment logs and use
the cluster's accounting command according to local policy.
