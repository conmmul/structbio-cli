# structbio

`structbio` gives structural-biology software the same kind of short, uniform
commands that SBGrid gives its own tools, for programs the lab's SBGrid CLI does
not cover. It is built for the GPU workstations in the lab: you name the run
type, a size, and an output folder, and the results appear in that folder.

```bash
rfdiffusion monomer 150 my_monomers -n 10
rfdiffusion binder target.pdb 100 my_binders --chain B --hotspots B30,B33
proteinmpnn design 7kdp.pdb 8 my_sequences --designable A:697-749
colabfold predict my_sequences my_folds --msa-mode single_sequence
cryozeta predict map.map.gz chains.fasta my_model --resolution 2.99 --contour 0.3
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

`structbio setup` scans the machine for software you already have, writes
`~/.config/structbio/config.yaml` with the paths it found, and installs one
short shell command per tool into `~/.local/bin`:

```text
Scanning for installed software...
  rfdiffusion    found      ~/software/RFdiffusion (environment: SE3nv)
  proteinmpnn    found      ~/software/ProteinMPNN (environment: mlfold)
  colabfold      found      /opt/localcolabfold/.pixi/envs/default/bin/colabfold_batch
  cryozeta       not found
Wrote ~/.config/structbio/config.yaml with 3 configured tool(s).
```

It looks on PATH, through your conda and pixi environments, and in the usual
software directories. `structbio detect` runs the same scan without writing
anything, and `structbio setup --update` adds newly found tools to a
configuration you already have, keeping a `.bak` copy and never changing an
entry you wrote yourself. Those commands are tiny
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
  colabfold:
    executable: colabfold_batch
    manager: none
```

`path` is the root of the upstream checkout and `executable` is relative to it.

### Installing a tool you do not have

```bash
structbio install rfdiffusion --into ~/software
```

This clones the project, records its path in your configuration, and then
prints that project's own remaining setup steps — creating the conda
environment and downloading the model weights — for you to run.

`structbio` deliberately stops before those steps. They differ per machine,
change with each upstream release, and in CryoZeta's case the weights are
licensed for academic and non-commercial use only, which is not a decision a
wrapper should make for you. `--dry-run` shows the whole plan, including the
licence, without cloning anything.

Then confirm what the workstation can reach:

```bash
structbio doctor
structbio tools
structbio config
```

If a tool reports a missing or mismatched PyTorch, `structbio fix-env` works
out the build this machine's driver needs and prints the command; `--run`
installs it.

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
| `colabfold predict SEQUENCES OUTPUT` | Fold sequences, or a whole ProteinMPNN run |
| `cryozeta predict MAP CHAINS.fasta OUTPUT` | Model a structure into a cryo-EM map |
| `cryozeta predict-json TARGETS.json OUTPUT` | CryoZeta run from a hand-written target JSON |

Options shared by every quick command:

- `-n, --num` designs or models to produce.
- `--gpu 0`, `--gpu 0,1`, or `--gpu auto` to take the card with the most free
  memory — useful when someone else is already using the workstation.
- `--dry-run` prints the exact upstream command and creates nothing.
- `--quiet` stops the tool's output being echoed; it is still logged.
- `--set dotted.key=value` reaches any option the short form does not expose,
  for example `--set diffusion.timesteps=50`.

The wrapped tool's output is streamed to your terminal as it runs, so a long
job shows its progress instead of going silent for an hour. Everything is
written to the run's log files either way.

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

`colabfold predict` warns you when a run would send sequences to the public
MMseqs2 server, which the default MSA mode does. `--msa-mode single_sequence`
keeps unpublished sequences on the machine.

`cryozeta predict` writes CryoZeta's native target JSON for you from the map and
a FASTA of the chains, so `--resolution` and `--contour` are the only extra
things to supply. It reads the map header before starting, counts identical
chains as copies, and steers complexes above ~2800 residues to `--large`. It
will not guess whether an `ACGT`-only sequence is DNA or a peptide — say which
with `--dna`, `--rna`, or `--protein`.

## Chaining a design run

Each stage reads the previous stage's output folder, so the usual design
workflow is three commands:

```bash
rfdiffusion monomer 150 my_backbones -n 20
proteinmpnn design my_backbones 4 my_sequences
colabfold predict my_sequences my_folds --msa-mode single_sequence
```

`proteinmpnn design` accepts a folder of backbones, and `colabfold predict`
finds the designed sequences inside a ProteinMPNN output folder by itself,
whether that run covered one structure or a whole batch.
Compare each folded model against the backbone it came from to decide which
designs are worth making.

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
| `structbio detect` | no | no |
| `structbio install TOOL` | clones a project | no |
| `structbio setup` / `doctor` / `tools` / `config` | configuration only | no |

## Configuration precedence

Values merge in this order, with later layers winning: package defaults,
optional lab-wide config (`/etc/structbio/config.yaml` or
`STRUCTBIO_LAB_CONFIG`), user config (`~/.config/structbio/config.yaml` or
`STRUCTBIO_USER_CONFIG`), then either the experiment YAML or the arguments typed
on the command line, and finally `--set` overrides. No workstation path is
hardcoded.

## More help

- [**PROTOCOL.md**](PROTOCOL.md) — a step-by-step guide that assumes no
  command-line experience. Start here if the terminal is new to you, and give
  this to new lab members.
- [Copyable wrapper configuration catalog](examples/README.md)
- [Installation and environments](docs/installation.md)
- [RFdiffusion wrapper: quick commands, modes, and YAML fields](docs/rfdiffusion.md)
- [ProteinMPNN wrapper: mutation masks, constraints, and batching](docs/proteinmpnn.md)
- [ColabFold wrapper: folding designs, and keeping sequences local](docs/colabfold.md)
- [CryoZeta wrapper: maps, chains, and the large-complex pipeline](docs/cryozeta.md)
- [Adding a backend](docs/architecture.md)
- [Troubleshooting](docs/troubleshooting.md)
- [Optional: shared clusters and SLURM](docs/cluster.md)
- [Verification checklist before rolling this out](docs/verification.md)
