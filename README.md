# structbio

`structbio` is a shared command-line interface for scientific software that is
not currently covered by the lab's SBGrid CLI. It is designed for researchers
who usually run these programs on HPC systems rather than local workstations.
Compared with thin Colab interfaces, it provides more control, reproducible
configuration, explicit resource requests, and persistent run records. It wraps
existing installations rather than reimplementing the scientific programs, and
additional wrappers can be added over time.

## First-time setup

Python 3.10 or newer is required. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
structbio doctor
```

Create `~/.config/structbio/config.yaml`:

```yaml
tools:
  rfdiffusion:
    environment: SE3nv
    manager: conda
    path: /work/software/RFdiffusion
    executable: scripts/run_inference.py
  proteinmpnn:
    environment: SE3nv
    manager: conda
    path: /work/software/ProteinMPNN
    executable: protein_mpnn_run.py
  cryozeta:
    environment: default
    manager: pixi
    path: /work/software/CryoZeta
    executable: inference_demo.sh
```

The configured `path` is the root of the upstream tool checkout, and
`executable` is relative to that root. `structbio` does not install model
weights or the scientific tools themselves.

Check the setup before preparing a run:

```bash
structbio doctor
structbio tools
```

`doctor` reports missing optional software without failing. A wrapped tool is
ready only when it is shown as configured and available on the machine where
the command will run.

## Running a wrapper safely

Use the same review sequence for every scientific tool:

1. Copy the closest YAML file from `examples/` into your own working directory.
2. Edit the experiment name, input paths, design settings, and resources.
3. Validate the YAML and all referenced structure selections.
4. Print and review the exact upstream command.
5. Perform a dry run to review the experiment and output paths.
6. Run locally, or generate and explicitly submit a SLURM script.

For RFdiffusion:

```bash
cp examples/rfdiffusion/tetrahedral.yaml tetrahedral.yaml
structbio rfdiffusion validate tetrahedral.yaml
structbio rfdiffusion command tetrahedral.yaml
structbio rfdiffusion run tetrahedral.yaml --dry-run
structbio rfdiffusion run tetrahedral.yaml
```

For ProteinMPNN, always inspect the mutation mask before the dry run:

```bash
cp examples/proteinmpnn/design_region.yaml mpnn.yaml
# Edit input.pdb and the requested chains/residues first.
structbio proteinmpnn validate mpnn.yaml
structbio proteinmpnn inspect-mask mpnn.yaml
structbio proteinmpnn command mpnn.yaml
structbio proteinmpnn run mpnn.yaml --dry-run
structbio proteinmpnn run mpnn.yaml
```

For CryoZeta, prepare its native JSON first and point the wrapper YAML at it:

```bash
cp examples/cryozeta/dataset.yaml cryozeta.yaml
structbio cryozeta validate cryozeta.yaml
structbio cryozeta command cryozeta.yaml
structbio cryozeta run cryozeta.yaml --dry-run
structbio cryozeta run cryozeta.yaml
```

Remove `--dry-run` only after reviewing the command and output location. Every
real run receives a new directory below `experiments/`; an existing experiment
is never overwritten.

## What each command does

| Command | Creates an experiment? | Runs science software? | Submits a job? |
| --- | --- | --- | --- |
| `structbio validate CONFIG.yaml` | no | no | no |
| `structbio TOOL validate CONFIG.yaml` | no | no | no |
| `structbio TOOL command CONFIG.yaml` | no | no | no |
| `structbio TOOL run CONFIG.yaml --dry-run` | no | no | no |
| `structbio TOOL run CONFIG.yaml` | yes | yes | no |
| `structbio TOOL submit CONFIG.yaml --dry-run` | no | no | no |
| `structbio TOOL submit CONFIG.yaml` | yes | no | no |
| `structbio TOOL submit CONFIG.yaml --execute` | yes | no locally | yes |
| `structbio proteinmpnn inspect-mask CONFIG.yaml` | no | no | no |
| `structbio status [EXPERIMENT_ID]` | no | no | no |
| `structbio tools` / `structbio doctor` | no | no | no |

`submit` writes a SLURM script but does not call `sbatch` unless `--execute` is
also given. This makes command generation safe on login nodes and in tests.
Use repeatable `--set dotted.key=value` options for temporary overrides, for
example `--set resources.gpus=4 --set design.num_designs=100`.

## Finding the results

A real run or prepared submission creates a directory such as:

```text
experiments/tetra600_2026-08-28_001/
├── config.yaml       # fully merged configuration used for the run
├── command.txt       # exact wrapped command
├── metadata.json     # tool, host, versions, inputs, outputs, and job information
├── environment.txt   # Python, Conda, CUDA, GPU, and SLURM environment
├── stdout.log
├── stderr.log
├── job.slurm         # present for prepared/submitted SLURM jobs
├── inputs/           # generated sidecar files; raw source data is not changed
├── outputs/          # wrapped tool output
└── analysis/         # reserved for downstream analysis
```

Use `structbio status` to list all recorded experiments, or
`structbio status EXPERIMENT_ID` for one exact directory name. If a SLURM job ID
is recorded and `squeue` is available, its scheduler state is shown too.

## Configuration precedence

Values merge in this order, with later layers winning: package defaults,
lab-wide config (`/etc/structbio/config.yaml` or `STRUCTBIO_LAB_CONFIG`), user
config (`~/.config/structbio/config.yaml` or `STRUCTBIO_USER_CONFIG`), experiment
YAML, then command-line overrides when a command supplies them. No cluster name
or software path is hardcoded.

## More help

- [Copyable wrapper configuration catalog](examples/README.md)
- [Installation and environments](docs/installation.md)
- [HPC and SLURM](docs/hpc.md)
- [RFdiffusion wrapper: modes, YAML fields, and examples](docs/rfdiffusion.md)
- [ProteinMPNN wrapper: mutation masks, constraints, and batching](docs/proteinmpnn.md)
- [CryoZeta wrapper: native JSON and inference modes](docs/cryozeta.md)
- [Adding a backend](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
