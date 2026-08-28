# structbio

`structbio` is a  command-line
interface for installed scientific software. This is specifically for software not currently adopted by SBGrid-cli. Many people in my lab don't run these on workstations locally, but it is a better option than using the thin colabs that are sometimes included. Hopefully this is an easier way to run these programs with more control. Will keep updating softwares.

## Install

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

Then validate before consuming GPU time:

```bash
structbio rfdiffusion run examples/rfdiffusion/tetrahedral.yaml --dry-run
structbio proteinmpnn inspect-mask examples/proteinmpnn/design_region.yaml
structbio doctor
```

Remove `--dry-run` only after reviewing the command and output location. Every
real run receives a new directory below `experiments/`; an existing experiment
is never overwritten.

## Common commands

```bash
structbio validate CONFIG.yaml
structbio rfdiffusion command CONFIG.yaml
structbio rfdiffusion run CONFIG.yaml [--dry-run]
structbio rfdiffusion submit CONFIG.yaml [--dry-run] [--execute]
structbio proteinmpnn inspect-mask CONFIG.yaml
structbio cryozeta validate CONFIG.yaml
structbio status [EXPERIMENT_ID]
structbio tools
structbio doctor
```

`submit` writes a SLURM script but does not call `sbatch` unless `--execute` is
also given. This makes command generation safe on login nodes and in tests.
Use repeatable `--set dotted.key=value` options for temporary overrides, for
example `--set resources.gpus=4 --set design.num_designs=100`.

## Configuration precedence

Values merge in this order, with later layers winning: package defaults,
lab-wide config (`/etc/structbio/config.yaml` or `STRUCTBIO_LAB_CONFIG`), user
config (`~/.config/structbio/config.yaml` or `STRUCTBIO_USER_CONFIG`), experiment
YAML, then command-line overrides when a command supplies them. No cluster name
or software path is hardcoded.

## More help

- [Installation and environments](docs/installation.md)
- [HPC and SLURM](docs/hpc.md)
- [RFdiffusion](docs/rfdiffusion.md)
- [ProteinMPNN](docs/proteinmpnn.md)
- [CryoZeta](docs/cryozeta.md)
- [Adding a backend](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
