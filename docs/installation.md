# Installation and environments

Install `structbio` in its own lightweight Python environment. Wrapped tools
remain in their vendor-recommended environments; `structbio` uses `conda run -n`
instead of relying on an activated interactive shell.

## 1. Install the toolkit

```bash
git clone https://github.com/conmmul/structbio-cli.git
cd structbio-cli
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
structbio --help
```

An editable installation is useful for a shared lab checkout because pulling an
update immediately updates the installed command. Keep the virtual environment
activated when using `structbio`, or call `.venv/bin/structbio` directly.

## 2. Configure installed scientific tools

Put per-user installation paths in `~/.config/structbio/config.yaml`:

```yaml
tools:
  rfdiffusion:
    path: /work/software/RFdiffusion
    executable: scripts/run_inference.py
    manager: conda
    environment: SE3nv

  proteinmpnn:
    path: /work/software/ProteinMPNN
    executable: protein_mpnn_run.py
    manager: conda
    environment: proteinmpnn

  cryozeta:
    path: /work/software/CryoZeta
    executable: inference_demo.sh
    manager: pixi
    environment: default
```

Field meanings:

| Field | Meaning |
| --- | --- |
| `path` | Absolute path to the root of the upstream tool checkout. |
| `executable` | Executable or script path, relative to `path`. |
| `manager` | `conda`, `pixi`, or `none`. |
| `environment` | Conda/Pixi environment name used by that installation. |

RFdiffusion and ProteinMPNN commands are wrapped with `conda run -n ENV` when a
Conda manager and environment are configured. CryoZeta's verified upstream
script manages Pixi itself; the configured environment is passed as its
`--env` argument.

## 3. Use a shared lab configuration

A lab manager can keep the same settings in a shared read-only file:

```bash
export STRUCTBIO_LAB_CONFIG=/work/lab/config/structbio.yaml
```

Users can still override selected values in their own
`~/.config/structbio/config.yaml`. See `examples/lab-config.yaml` for tool and
cluster-profile settings in one file.

## 4. Verify the machine

Run these commands on the workstation or cluster login node where jobs will be
prepared:

```bash
structbio doctor
structbio tools
```

For each wrapped tool, verify that:

- the path is the intended upstream checkout;
- the configured executable exists below that path;
- Conda or Pixi is available as appropriate;
- model weights and upstream dependencies were installed according to the
  scientific tool's own documentation;
- input and output paths will also be visible from compute nodes.

`doctor` does not treat missing optional tools, GPUs, or SLURM as fatal. For
example, a login node may legitimately report no CUDA device even though its
submitted compute jobs receive GPUs.

Do not install RFdiffusion, ProteinMPNN, or CryoZeta merely by installing this
package. Follow each upstream project's instructions and license terms.
