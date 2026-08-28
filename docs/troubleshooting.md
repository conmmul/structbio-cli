# Troubleshooting

Start with `structbio doctor`, then re-run the command with `--dry-run` to see
exactly what would be executed.

## The GPU is not detected

`structbio gpu` reports what the machine said and why the answer failed, rather
than the bare "NOT FOUND" that `doctor` shows.

| Cause | Signature | Fix |
| --- | --- | --- |
| Not on PATH | `nvidia-smi was not found` | `STRUCTBIO_NVIDIA_SMI=/usr/bin/nvidia-smi` |
| Driver error | `exited with code N: ...` | The text after the code is the driver's own. |
| Slow or wedged driver | `did not answer within 20 seconds` | Run `nvidia-smi` yourself; the queries allow 20 seconds, not 3. |
| No cards listed | `ran but listed no GPUs` | Hardware or driver configuration. |

An old driver that cannot report `compute_cap` is **not** an absent GPU. The
capability is then inferred from the model name, `structbio gpu` says it did
so, and `--capability 8.9` states it outright when the card is not recognised.

## Which structbio am I running?

`structbio doctor` names it first:

```text
structbio              0.2.0
  package              /Users/you/structbio-cli/src/structbio
  interpreter          /Users/you/structbio-cli/.venv/bin/python
```

### No such command 'install'. Did you mean 'install-wrappers'?

An older checkout is being run. More than one clone of this repository, each
with its own virtual environment, is the usual cause: whichever environment is
active wins, regardless of which clone you are working in. Compare the package
path in `structbio doctor` with the checkout you expect, and either activate the
right environment or delete the stale clone.

`structbio install-wrappers` writes a `structbio` command of its own into
`~/.local/bin`, alongside the per-tool ones. It names its interpreter outright,
so it runs the installation it was generated from whatever environment happens
to be active. Re-run it after moving or rebuilding an environment.

## Installation diagnostics

### NOT CONFIGURED

Try `structbio detect` first: if the tool is installed somewhere it looks,
`structbio setup --update` will record it for you. If detection misses it,
the tool is either outside the searched directories, or its directory does not
contain the entry-point script that identifies it — a `RFdiffusion` folder
without `scripts/run_inference.py` is deliberately ignored.

Otherwise add the tool path, executable, manager, and environment to the user
configuration; `structbio config` prints which file is in use, and
`structbio setup` creates it from a template. Confirm that `path/executable`
names the actual installed script, not an example command from another checkout
version.

### CONFIGURED, UNAVAILABLE

`structbio doctor` names the specific reason and the fix beneath the status
line. The distinctions it makes:

| Reason | Meaning |
| --- | --- |
| `the configured path does not exist` | Nothing is installed there. The default configuration lists every tool as a starting point, so an entry does not imply an installation. |
| `the configured path is not a folder` | The `path` value points at a file. |
| `... exists but does not contain EXECUTABLE` | Incomplete clone, or a different project, or `executable` is wrong for this version. |
| `EXECUTABLE is not on PATH, and no path is configured` | For tools installed as a command rather than a checkout, such as ColabFold. |
| `conda is not installed` | The entry asks for a conda environment on a machine without conda. Set `manager: none` if no environment is needed. |
| `the conda environment 'X' does not exist` | The code is present but its environment was never created; the install steps were probably stopped partway. |
| `pixi is not installed` | Needed by CryoZeta. |
| `PyTorch is not installed in the conda environment 'X'` | The environment exists but the tool's main dependency is absent. |
| `PyTorch ... is a CPU-only build` | A warning, not a failure: it runs, ignoring the GPU. The tool reports `FOUND, DEGRADED`. |
| `PyTorch ... was built for CUDA X, which this driver cannot run` | Fatal: the wheel is newer than the driver and fails at import. |

### PyTorch

`structbio fix-env` reads the CUDA version the NVIDIA driver reports, picks the
newest PyTorch wheel index at or below it, and prints the install command.
`--run` executes it after asking; `--force` replaces a PyTorch that is already
there.

The build is read from the environment's `torch/version.py` rather than by
importing torch, so the check costs nothing and works even when the
installation is too broken to import.

The choice is deliberately conservative: a wheel built for a newer CUDA than
the driver will not run, so a driver reporting 12.0 is given `cu118` rather
than `cu121`.

### Environments a project pins

RFdiffusion's `env/SE3nv.yml` fixes `pytorch=1.9` with `cudatoolkit=11.1` and
`dgl-cuda11.1`, and lists conda-forge ahead of the pytorch channel, so conda
readily resolves a CPU-only build there. Installing a current PyTorch to fix
that breaks the checkout, because SE3Transformer is built against the pinned
version.

`structbio fix-env` therefore refuses to touch such an environment and prints
the repair the project's own pins imply:

```bash
conda install -n SE3nv -c pytorch -c nvidia pytorch=1.9 cudatoolkit=11.1
```

Rebuilding the environment from the file is the alternative. A backend declares
this by setting `pinned_environment`; without it, a plain PyTorch install is
offered instead.

### When the GPU is newer than the pinned CUDA

Some mismatches cannot be fixed by installing anything. A PyTorch built against
an old CUDA contains no machine code for a newer GPU architecture, and fails
with `no kernel image is available for execution on the device`. structbio
reads each card's compute capability from `nvidia-smi` and says so outright
rather than offering a repair that cannot work:

```text
this machine's GPU is Ada Lovelace, such as the RTX 40 series and L40, which
needs CUDA 11.8 or newer, but RFdiffusion's env/SE3nv.yml pins CUDA 11.1.
No PyTorch install can bridge that: the pinned version has no kernels for
this card
```

| Architecture | Compute capability | Oldest CUDA that supports it |
| --- | --- | --- |
| Ampere datacenter (A100) | 8.0 | 11.0 |
| Ampere (A6000, RTX 30) | 8.6 | 11.1 |
| Ada Lovelace (RTX 40, L40) | 8.9 | 11.8 |
| Hopper (H100) | 9.0 | 11.8 |
| Blackwell (RTX 50, B-series) | 10.0, 12.0 | 12.8 |

RFdiffusion pins CUDA 11.1 in `env/SE3nv.yml`, and its Dockerfile uses CUDA
11.6, so **neither supports Ada, Hopper, or Blackwell**. On those cards the
options are to build an environment with a current PyTorch and a matching DGL,
which is beyond what this wrapper can promise, or to run on an older card.
Raise it upstream before assuming the rest of the checkout works against a
newer PyTorch.

A tool that lives on a shared filesystem is only usable from a machine that
mounts it. If a tool is installed somewhere structbio does not know about,
`structbio detect` followed by `structbio setup --update` is quicker than
editing the configuration.

To stop a tool being reported at all, delete its entry from the user
configuration.

### structbio install stopped after cloning

That is what it does. Environment creation and weight downloads are left to the
upstream project's own instructions, which the command prints. Run those, then
`structbio setup --update` and `structbio doctor`.

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

## ColabFold errors

- **Sequences leave the machine:** every `mmseqs2_*` MSA mode queries the public
  ColabFold server. Use `--msa-mode single_sequence`, or set `msa.host_url` to
  your own server, for unpublished sequences.
- **Non-amino-acid characters:** the FASTA holds something other than a
  sequence. Check for a stray header, a numeric column, or a pasted PDB line.
- **Duplicate job names:** two records share a name, so their results overwrite
  each other. Rename the records.
- **Nothing found in a folder:** ColabFold reads `.fa`, `.fasta`, `.a3m`,
  `.csv`, and `.tsv`. A ProteinMPNN output folder works because its sequences
  are in `seqs/`.
- **The first run stalls:** ColabFold is downloading AlphaFold parameters into
  its own cache. That happens once.

## CryoZeta errors

- **Not an MRC/CCP4 density map:** the file has no `MAP ` stamp at byte 208.
  Check for a truncated download, or a `.pdb` or `.mrcs` given by mistake.
- **Cannot tell whether a chain is protein, DNA, or RNA:** the sequence uses
  only A, C, G and T, which are valid in both. Say which with `--dna`, `--rna`,
  or `--protein`, or with `input.chains` in YAML.
- **A complex above ~2800 residues:** add `--large` to use
  `large_inference_demo.sh`. Its options differ: `--registration` instead of
  `--mode`, and `--detection-checkpoint` instead of `--interp-checkpoint`.
- **Ligands, ions, modifications, or several targets:** the short command cannot
  express them. Write CryoZeta's own JSON and use `cryozeta predict-json`.
- Prefer absolute map and MSA paths so they do not depend on the current
  directory.
- Confirm Pixi assets, checkpoints, CUDA toolchain, and compiler cache locations.
- The standard wrapped pipeline is single-GPU; requesting extra GPUs does not
  parallelize it.

## Watching a run

Wrapped tool output is streamed to the terminal as it arrives and written to the
log at the same time. If a run looks silent:

- pass `--dry-run` first to confirm the command is what you expect;
- check `OUTPUT/.structbio/stdout.log` and `stderr.log`, which have everything
  even when `--quiet` was used;
- remember that a tool doing MSA or parameter downloads can be genuinely quiet
  for several minutes before it prints anything.

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
