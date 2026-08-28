# structbio

`structbio` gives structural-biology software the same kind of short, uniform
commands that SBGrid gives its own tools, for programs the lab's SBGrid CLI does
not cover. It is built for the GPU workstations in the lab: you name the run
type, a size, and an output folder, and the results appear in that folder.

```bash
rfdiffusion monomer 150 my_monomers -n 10
rfdiffusion binder target.pdb 100 my_binders --chain B --hotspots B30,B33
proteinmpnn design 7kdp.pdb 8 my_sequences --designable A:697-749
cryozeta predict targets.json my_maps --gpu 0
```

Each command wraps an existing installation instead of reimplementing the
science, checks the request against the actual input structure before anything
runs, and records exactly what was executed next to the results. More wrappers
can be added over time.

## First-time setup

Python 3.10 or newer is required. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
structbio setup
```

`structbio setup` writes `~/.config/structbio/config.yaml` and installs one
short shell command per tool into `~/.local/bin`. Those commands are tiny
wrappers: `rfdiffusion ...` is exactly `structbio rfdiffusion ...`. The same
wrappers are checked into [`bin/`](bin) if you would rather copy them into a
shared directory yourself.

If `setup` reports that `~/.local/bin` is not on your PATH, do what it says
before going further, or the shell will not find the commands:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Edit the configuration so every path points at software that is already
installed on the workstation:

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
    environment: mlfold
```

`path` is the root of the upstream checkout and `executable` is relative to it.
`structbio` never installs the scientific software or its model weights.

Then confirm what the workstation can reach:

```bash
structbio doctor
structbio tools
structbio config
```

A tool is ready only when `doctor` reports it as `FOUND`.

## Quick commands

The last argument is always the output folder, and its name becomes the prefix
of the files inside it, so `rfdiffusion monomer 150 my_monomers` writes
`my_monomers/my_monomers_0.pdb` and so on.

| Command | What it does |
| --- | --- |
| `rfdiffusion monomer LENGTH OUTPUT` | Unconditional monomer backbones |
| `rfdiffusion symmetry GROUP TOTAL_LENGTH OUTPUT` | Symmetric oligomers (`c4`, `d3`, `tetrahedral`) |
| `rfdiffusion binder TARGET.pdb LENGTH OUTPUT` | Binders against a target chain |
| `rfdiffusion partial INPUT.pdb STEPS OUTPUT` | Diversify an existing structure |
| `proteinmpnn design INPUT NUM_SEQUENCES OUTPUT` | Sequences for a PDB, or a folder of PDBs |
| `cryozeta predict TARGETS.json OUTPUT` | CryoZeta inference on its own target JSON |

Options shared by every quick command:

- `-n, --num` designs to produce (RFdiffusion).
- `--gpu 0` or `--gpu 0,1` picks the GPU on this workstation.
- `--dry-run` prints the exact upstream command and creates nothing.
- `--set dotted.key=value` reaches any option the short form does not expose,
  for example `--set diffusion.timesteps=50`.

Run `rfdiffusion --help`, or `rfdiffusion binder --help`, for the full list.

Two things are always checked before a tool starts. Residue selections and
hotspots must exist in the input structure, and the output folder must be new or
empty: `structbio` never writes over previous results.

### Worth knowing per tool

`rfdiffusion binder` reads the target contig out of the residue numbering in the
file itself, so you only give it a chain. Name the chain with `--chain` whenever
the file holds more than one.

`rfdiffusion symmetry` takes the total length across all subunits, and refuses a
length that does not divide by the subunit count.

`proteinmpnn design` lets every residue of the selected chains change unless you
restrict it with `--designable`, and it prints the mutable and fixed residues,
in the original PDB numbering, before it runs.

## Results and provenance

```text
my_monomers/
├── my_monomers_0.pdb      # wrapped tool output, at the top level
├── my_monomers_1.pdb
└── .structbio/            # what produced them
    ├── config.yaml        # fully merged configuration
    ├── command.txt        # exact command that ran
    ├── metadata.json      # tool, host, versions, commits, inputs, outputs
    ├── environment.txt    # Python, Conda, CUDA, and GPU environment
    ├── stdout.log
    └── stderr.log
```

`structbio status my_monomers` prints that record back.

## Full control with YAML

Quick commands cover the common runs. Everything the wrapped tools document —
motif scaffolding, guiding potentials, per-position biases, batching — lives in
a YAML configuration instead:

```bash
cp examples/rfdiffusion/tetrahedral.yaml tetra.yaml
structbio rfdiffusion validate tetra.yaml     # check the YAML and the selections
structbio rfdiffusion command tetra.yaml      # print the upstream command
structbio rfdiffusion run tetra.yaml --dry-run
structbio rfdiffusion run tetra.yaml
```

For ProteinMPNN, review the mutation mask first:

```bash
structbio proteinmpnn inspect-mask mpnn.yaml
```

A YAML run writes to a dated folder under `experiments_root` rather than to a
folder you name, and `structbio status` lists those runs.

## What each command does

| Command | Creates a folder? | Runs science software? |
| --- | --- | --- |
| `TOOL RUNTYPE ... OUTPUT` | yes, the folder you named | yes |
| `TOOL RUNTYPE ... OUTPUT --dry-run` | no | no |
| `structbio TOOL run CONFIG.yaml` | yes, under `experiments_root` | yes |
| `structbio TOOL run CONFIG.yaml --dry-run` | no | no |
| `structbio TOOL validate CONFIG.yaml` | no | no |
| `structbio TOOL command CONFIG.yaml` | no | no |
| `structbio proteinmpnn inspect-mask CONFIG.yaml` | no | no |
| `structbio status [FOLDER \| EXPERIMENT_ID]` | no | no |
| `structbio setup` / `doctor` / `tools` / `config` | configuration only | no |

## Configuration precedence

Values merge in this order, with later layers winning: package defaults,
optional lab-wide config (`/etc/structbio/config.yaml` or
`STRUCTBIO_LAB_CONFIG`), user config (`~/.config/structbio/config.yaml` or
`STRUCTBIO_USER_CONFIG`), then either the experiment YAML or the arguments typed
on the command line, and finally `--set` overrides. No workstation path is
hardcoded.

## More help

- [Copyable wrapper configuration catalog](examples/README.md)
- [Installation and environments](docs/installation.md)
- [RFdiffusion wrapper: quick commands, modes, and YAML fields](docs/rfdiffusion.md)
- [ProteinMPNN wrapper: mutation masks, constraints, and batching](docs/proteinmpnn.md)
- [CryoZeta wrapper: native JSON and inference modes](docs/cryozeta.md)
- [Adding a backend](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Optional: shared clusters and SLURM](docs/cluster.md)
