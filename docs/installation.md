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

`structbio setup` scans for software you already have and writes
`~/.config/structbio/config.yaml` from what it finds. It looks, in this order:

1. **PATH**, for tools installed as a command, such as `colabfold_batch`.
2. **Conda and pixi environments**, from `conda env list`, both for executables
   inside them and for environment names a tool conventionally uses (`SE3nv`
   for RFdiffusion, `mlfold` for ProteinMPNN). An environment name is only
   taken when it belongs to that tool: borrowing another tool's environment
   would put a wrong `conda run -n` in front of every command.
3. **The usual software directories** — `~/software`, `~/apps`, `~/src`,
   `~/opt`, `~/tools`, your home directory, `/opt`, and `/usr/local` — two
   levels deep, looking for a directory named after the tool that actually
   contains its entry-point script. A directory named `RFdiffusion` with no
   `scripts/run_inference.py` in it is ignored rather than configured wrongly.

Related commands:

```bash
structbio detect          # run the scan, change nothing
structbio setup --update  # add newly found tools to an existing config
structbio setup --no-detect  # write the plain template instead
```

`--update` keeps a `.bak` copy of the previous file and never changes an entry
you wrote yourself; it only adds tools that are missing.

The result looks like this, and you can edit it freely:

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

## 5. Installing the tools themselves

```bash
structbio install rfdiffusion --into ~/software
structbio install proteinmpnn --into ~/software
structbio install colabfold   --into ~/software
structbio install cryozeta    --into ~/software
```

Each of these clones the project, records the path in your configuration, and
prints that project's own remaining steps, read from its README on the date the
command shows. `--dry-run` prints the whole plan, including the licence, and
clones nothing.

`structbio` clones and stops. It does not create conda or pixi environments and
does not download model weights, because:

- the right PyTorch or CUDA build depends on the machine, and upstream projects
  point at pytorch.org rather than pinning one;
- those steps change with each release, and a copy of them here would rot
  silently while looking authoritative;
- weights carry their own terms. CryoZeta's are free for academic and
  non-commercial research use only, and accepting that is yours to do, not a
  wrapper's.

What each project still needs after the clone:

| Tool | Remaining work | Weights |
| --- | --- | --- |
| RFdiffusion | `conda env create -f env/SE3nv.yml`, install SE3Transformer, `pip install -e .` | 7 checkpoints from files.ipd.uw.edu |
| ProteinMPNN | A conda environment with PyTorch | Already in the repository |
| ColabFold | `pixi install && pixi run setup` in the localcolabfold checkout | AlphaFold parameters fetched on first run |
| CryoZeta | `pixi run setup` | From Hugging Face, non-commercial licence |

Afterwards, `structbio setup --update` records anything new and
`structbio doctor` confirms it is reachable.
