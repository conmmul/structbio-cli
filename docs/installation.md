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
structbio setup
```

An editable installation is useful for a shared lab checkout because pulling an
update immediately updates the installed command.

`structbio setup` also writes one short command per tool into `~/.local/bin`, so
`rfdiffusion monomer 150 my_monomers` works from any directory. Each of those
files is a five-line shell wrapper around `structbio`, and it records the
absolute path of the interpreter it was generated from, so the virtual
environment does not have to be activated first. Re-run
`structbio install-wrappers` after moving or rebuilding the environment. If
`~/.local/bin` is not on your PATH yet:

```bash
eval "$(structbio shell-init)"
```

Add that line to `~/.zshrc` or `~/.bashrc` to make it permanent; a fresh macOS
account does not have `~/.local/bin` on PATH, so this step is usually needed.
`structbio setup` checks and prints the exact line if it is missing. Until it is
done, `rfdiffusion` reports `command not found` even though the wrapper exists. To place the
commands somewhere else, such as a shared directory for the whole lab, pass
`--bin-dir`:

```bash
structbio install-wrappers --bin-dir /usr/local/structbio/bin
```

## 2. Configure installed scientific tools

`structbio setup` creates `~/.config/structbio/config.yaml` from a template.
Edit it so every path points at software already installed on this machine:

```yaml
tools:
  rfdiffusion:
    path: ~/software/RFdiffusion
    executable: scripts/run_inference.py
    manager: conda
    environment: SE3nv

  proteinmpnn:
    path: ~/software/ProteinMPNN
    executable: protein_mpnn_run.py
    manager: conda
    environment: proteinmpnn

  # ColabFold usually installs colabfold_batch onto PATH.
  colabfold:
    executable: colabfold_batch
    manager: none

  cryozeta:
    path: ~/software/CryoZeta
    executable: inference_demo.sh
    manager: pixi
    environment: default
```

Field meanings:

| Field | Meaning |
| --- | --- |
| `path` | Path to the root of the upstream tool checkout; `~` is expanded. |
| `executable` | Executable or script path, relative to `path`. |
| `manager` | `conda`, `pixi`, or `none`. |
| `environment` | Conda/Pixi environment name used by that installation. |

`path` may be omitted for a tool whose executable is already on PATH, as
ColabFold's usually is; the executable is then looked up there.

RFdiffusion, ProteinMPNN, and ColabFold commands are wrapped with
`conda run -n ENV` when a Conda manager and environment are configured. CryoZeta's verified upstream
script manages Pixi itself; the configured environment is passed as its
`--env` argument.

## 3. Use a shared lab configuration

A lab manager can keep the same settings in a shared read-only file:

```bash
export STRUCTBIO_LAB_CONFIG=/work/lab/config/structbio.yaml
```

Users can still override selected values in their own
`~/.config/structbio/config.yaml`. See `examples/lab-config.yaml` for tool
settings in one file.

## 4. Verify the machine

Run these commands on the workstation where the tools will run:

```bash
structbio doctor
structbio tools
structbio config
```

For each wrapped tool, verify that:

- the path is the intended upstream checkout;
- the configured executable exists below that path;
- Conda or Pixi is available as appropriate;
- model weights and upstream dependencies were installed according to the
  scientific tool's own documentation;
- a GPU is visible, if the tool needs one.

`doctor` does not treat missing optional tools as fatal, and it reports SLURM
only as an optional extra: nothing in a workstation run needs it.

Do not install RFdiffusion, ProteinMPNN, or CryoZeta merely by installing this
package. Follow each upstream project's instructions and license terms.
