# ProteinMPNN wrapper

The ProteinMPNN wrapper designs sequences on existing PDB backbones. Its most
important job is to make “fixed” versus “designable” explicit and reviewable
before any sequence generation starts.

ProteinMPNN internally represents fixed positions as one-based ordinal indices
within each chain. Researchers specify selections using the PDB's original
chain IDs and residue numbers, for example `A:697-749`. `structbio` translates
between those systems and reconstructs the mutable set from the generated
fixed-position dictionary. If the reconstruction differs by even one residue—or
is the opposite selection—the run is aborted.

## Before the first run

Configure the upstream checkout and environment:

```yaml
tools:
  proteinmpnn:
    path: /work/software/ProteinMPNN
    executable: protein_mpnn_run.py
    manager: conda
    environment: proteinmpnn
```

The ProteinMPNN weights and PyTorch environment must already be installed. Run
`structbio doctor` on the workstation or cluster where the command will run.

## Required review sequence

Always include `inspect-mask` for ProteinMPNN:

```bash
structbio proteinmpnn validate mpnn.yaml
structbio proteinmpnn inspect-mask mpnn.yaml
structbio proteinmpnn command mpnn.yaml
structbio proteinmpnn run mpnn.yaml --dry-run
```

The inspection reports the original PDB labels, for example:

```text
ProteinMPNN validation

Structure: tiny.pdb

Design chains: A

Designable residues:
A697-A699

Designable count: 3

Fixed residues:
A700, B10-B12

Fixed count: 4

Mask inversion check: PASSED
```

Proceed only if the listed designable residues are exactly the residues allowed
to mutate.

## Selection rules

The rules are intentionally simple:

| YAML selection | Result |
| --- | --- |
| No `design.chains` and no position lists | All chains and all residues are designable. |
| `design.chains: [A]` only | Every residue in A is designable; every other chain is fixed. |
| `design.chains: [A]` plus `fixed_positions` | Chain A is designable except those fixed positions. |
| `design.chains: [A]` plus `designable_positions` | Only the listed A positions are designable. |
| Both position lists | They must not overlap; overlap aborts validation. |

Positions use `CHAIN:START-END`, such as `A:697-749`, and ranges are inclusive.
Single positions use `A:697`. Insertion codes are supported for single
selections such as `A:10A`; ranges containing insertion codes are rejected
rather than silently reinterpreted.

## Design an entire PDB

```yaml
tool: proteinmpnn

experiment:
  name: all_chain_design

input:
  pdb: ../inputs/backbone.pdb

design:
  num_sequences: 8
  temperatures: [0.1]
  batch_size: 1
  seed: 37

resources:
  gpus: 1
```

With no chain or position restrictions, all parsed chains are passed as design
chains.

## Design one chain and fix the rest

```yaml
tool: proteinmpnn

experiment:
  name: chain_a_design

input:
  pdb: ../inputs/complex.pdb

design:
  chains: [A]
  num_sequences: 16
  temperatures: [0.1, 0.2]
  batch_size: 1
```

All residues in chain A are mutable. Every residue in every other chain is
placed in the fixed-position dictionary.

## Design only a residue range

```yaml
tool: proteinmpnn

experiment:
  name: design_a697_a749

input:
  pdb: ../inputs/7KDP.pdb

design:
  chains: [A]
  designable_positions: ["A:697-749"]
  num_sequences: 8
  temperatures: [0.1, 0.2]
  batch_size: 1
```

Quoting residue selections is recommended so YAML never interprets punctuation.
Only A697 through A749 may mutate.

## Design a chain except selected fixed residues

```yaml
design:
  chains: [A]
  fixed_positions:
    - "A:10-20"
    - "A:45"
  num_sequences: 8
  temperatures: [0.1]
  batch_size: 1
```

This is different from `designable_positions`: all of chain A is mutable except
A10-A20 and A45.

## Amino-acid constraints and biases

Global omissions and biases apply to every designable position:

```yaml
constraints:
  omit_aas: XCP
  bias_aas:
    A: -0.1
    G: 0.2
```

Position-specific settings use original PDB selections:

```yaml
constraints:
  omit_aas: X
  omit_by_position:
    "A:697-705": CP
    "A:720": W
  bias_by_position:
    "A:730-735":
      G: 0.5
      P: -0.5
```

Amino-acid codes are validated against ProteinMPNN's alphabet. Positive and
negative bias values change sampling preference; they do not guarantee or ban a
residue unless an omission is used.

## Soluble weights, sampling, and batches

```yaml
design:
  num_sequences: 32
  temperatures: [0.1, 0.15, 0.2]
  batch_size: 4
  seed: 37
  soluble_model: true
  model_name: v_48_020
```

- `num_sequences` is the number generated per input backbone and must be
  divisible by `batch_size` because that is how upstream ProteinMPNN schedules
  its batches.
- Higher temperatures generally produce more sequence diversity.
- `seed: 0` asks upstream ProteinMPNN to choose a random seed; use a nonzero seed
  for repeatable sampling.
- `soluble_model: true` passes the upstream soluble-weight flag. Confirm the
  configured checkout contains compatible soluble weights.

## Process a directory of PDBs

```yaml
tool: proteinmpnn

experiment:
  name: backbone_batch

input:
  directory: ../inputs/backbones

design:
  chains: [A]
  num_sequences: 8
  temperatures: [0.1]
  batch_size: 1

resources:
  gpus: 1
```

The directory is scanned for `.pdb` files in sorted filename order. Each PDB is
validated independently and receives its own command and output subdirectory.
The same requested chain and position selections must exist in every PDB; the
batch aborts rather than applying different interpretations to different files.

For an HPC array, first count and review the sorted PDB files, then set an array
whose zero-based indices match the generated command steps. See [HPC and
SLURM](hpc.md#slurm-arrays).

## Main YAML fields

| Field | Meaning |
| --- | --- |
| `input.pdb` | One PDB backbone. Mutually exclusive with `input.directory`. |
| `input.directory` | Directory of PDB backbones. |
| `design.chains` | Chains permitted to contain designable residues. |
| `design.designable_positions` | Exact original PDB positions allowed to mutate. |
| `design.fixed_positions` | Original PDB positions forced to remain fixed. |
| `design.num_sequences` | Sequences generated per backbone. |
| `design.temperatures` | One or more sampling temperatures greater than zero and at most one. |
| `design.batch_size` | Upstream ProteinMPNN batch size. |
| `design.seed` | Nonnegative upstream seed. |
| `design.soluble_model` | Select upstream soluble model weights. |
| `constraints.omit_aas` | Amino acids omitted globally. |
| `constraints.omit_by_position` | Amino acids omitted at selected PDB positions. |
| `constraints.bias_aas` | Global amino-acid sampling biases. |
| `constraints.bias_by_position` | Position-specific sampling biases. |

## Outputs and troubleshooting

For one PDB, ProteinMPNN writes below `EXPERIMENT/outputs/`. For a directory,
each PDB gets `EXPERIMENT/outputs/PDB_STEM/`. Generated fixed-position and bias
JSONL files are saved under `EXPERIMENT/inputs/` for auditability.

If validation fails, do not work around it by renumbering the raw PDB. Instead:

- confirm the requested chain IDs and residue numbers in the source structure;
- check for missing residues inside a requested range;
- use `inspect-mask` and compare designable and fixed counts;
- ensure `num_sequences` is divisible by `batch_size`;
- for a directory batch, verify that every PDB uses the same selection scheme.
