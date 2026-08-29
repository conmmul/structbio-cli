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

One command. Python 3.10 or newer is the only prerequisite.

```bash
git clone https://github.com/conmmul/structbio-cli.git
cd structbio-cli
./install.sh
```

`install.sh` creates a private Python environment, installs structbio into it,
and then runs `structbio setup`, which does everything else:

```text
structbio installer

Python            /usr/bin/python3.11 (3.11.9)
Environment       ~/structbio-cli/.venv (created)
Installing structbio...

Scanning for installed software...
  rfdiffusion    found      ~/software/RFdiffusion (environment: SE3nv)
  proteinmpnn    found      ~/software/ProteinMPNN (environment: mlfold)
  colabfold      found      /opt/localcolabfold/.pixi/envs/default/bin/colabfold_batch
  cryozeta       not found

Configuration     ~/.config/structbio/config.yaml (3 tool(s))
Commands          ~/.local/bin  (colabfold, cryozeta, proteinmpnn, rfdiffusion, structbio)
PATH              added to ~/.zshrc

Checking that each tool can run (this uses the GPU briefly)...
  rfdiffusion    ready            PyTorch 2.4.0 (CUDA 12.1), device: NVIDIA RTX 4090
  proteinmpnn    needs attention  PyTorch 1.9.1.post3 is installed but reports no usable GPU
                 fix: structbio env repair proteinmpnn

Open a new terminal, or run:  source ~/.zshrc

Ready to run: rfdiffusion

Try it:
  rfdiffusion monomer 100 my_first_designs -n 2
```

So `setup`:

- **finds the software you already have** — on PATH, through your conda and
  pixi environments, and in the usual software directories;
- **writes `~/.config/structbio/config.yaml`** with the paths it found, adding
  to that file on later runs and never changing an entry you wrote yourself;
- **installs one short command per tool** into `~/.local/bin`, so
  `rfdiffusion ...` is exactly `structbio rfdiffusion ...`;
- **puts those commands on your PATH**, choosing a shell start-up file you can
  actually write — a workstation built from a managed image often owns your
  `~/.zshrc`, and this picks `~/.zprofile` instead of failing;
- **proves each tool can run**, by running code on the GPU, and names the one
  command that fixes anything that cannot. An environment that already works
  is adopted rather than rebuilt.

Run `structbio setup` again whenever you install something new; it is safe to
repeat, and `--no-check`, `--no-path` or `--no-detect` switch off any part of
it. `structbio detect` scans without writing anything.

The commands each name their interpreter outright, so they run this
installation whatever virtual environment happens to be active — which matters
if more than one clone of this repository exists on the machine. Do not delete
`.venv`, and the same wrappers are checked into [`bin/`](bin) if you would
rather copy them into a shared directory yourself.

### You do not have to run setup first

If a command needs a tool that no configuration mentions, structbio looks for
it in the same places `setup` looks, uses what it finds, and records it so the
next run does not have to look again:

```text
$ rfdiffusion monomer 150 my_designs
Found rfdiffusion at ~/software/RFdiffusion (environment: SE3nv)
Recorded it in ~/.config/structbio/config.yaml
```

Only paths to software that is already installed are written. Nothing is
created, downloaded, or licence-accepted on your behalf. Set
`STRUCTBIO_NO_AUTOCONFIG=1` if you would rather configure everything by hand.

Then check what the workstation can reach:

```bash
structbio doctor
structbio gpu
```

### Getting the environments right

RFdiffusion and ProteinMPNN run from a Conda environment that has to contain a
PyTorch built for the GPU in the machine. That single requirement causes almost
all setup trouble, so take these in order.

`structbio setup` already checked each one and told you which of the following
you need. Nothing below is guesswork; each is the answer to a specific finding.

**1. `ready` — nothing to do.** If an environment already worked, setup adopted
it rather than building a second one beside it. To record one setup did not
look for, name it yourself:

```bash
structbio env adopt rfdiffusion --environment the_env_that_works
```

That runs code inside the environment and records it only if that succeeds.
Nothing is installed, changed or removed. It is always the better option: a
working environment took somebody hours.

**2. `needs attention` — repair the environment in place.**

```bash
structbio env verify rfdiffusion
```

```text
PyTorch 1.9.1.post3 (CUDA none), device: no GPU
FAILED: PyTorch 1.9.1.post3 is installed but reports no usable GPU
```

`CUDA none` means the installed PyTorch was built without CUDA at all. This is
the single most common problem: RFdiffusion's `env/SE3nv.yml` lists the general
conda channels ahead of the `pytorch` one, so conda readily picks a CPU build,
and a `.postN` version number is the tell-tale.

```bash
structbio env repair rfdiffusion
```

That replaces only PyTorch, and for RFdiffusion the matching DGL, leaving
everything else in the environment alone. It installs **from the conda
channels**, which matters: a machine may reach `conda.anaconda.org` while it
cannot reach `download.pytorch.org`, and a pip-based install then cannot work
at all.

**3. `no environment` — build one.**

```bash
structbio env create rfdiffusion
```

This chooses versions from the GPU actually present and refuses where no
working combination exists — an RTX 50-series card, for instance, needs CUDA
12.8 while DGL publishes nothing past 12.4.

`env create --force` never deletes anything: an existing environment is renamed
to `NAME-before-1` and the command tells you how to put it back.

### What to trust

`structbio env verify` is the authority, because it runs code and computes on
the GPU. Everything else structbio reports about versions comes from reading
files, and is advisory only — those readings have been wrong, so they warn and
never refuse. `doctor` showing `FOUND, WITH WARNINGS` means the tool works and
something may be slower than it needs to be.

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
| `structbio detect` / `structbio gpu` | no | no |
| `structbio install TOOL` | clones a project | no |
| `structbio env verify TOOL` | no | runs a short check on the GPU |
| `structbio env adopt TOOL` | no | runs a short check on the GPU |
| `structbio env repair TOOL` | changes one environment | no |
| `structbio env create TOOL` | builds an environment | no |
| `structbio setup` | configuration and PATH | runs each tool's check |
| `structbio doctor` / `tools` / `config` | no | no |

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
