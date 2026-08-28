# ColabFold wrapper

The ColabFold backend wraps the official `colabfold_batch` command, verified
against ColabFold 1.6.2. It predicts structures for sequences, which makes it
the usual last stage of a design run: RFdiffusion produces backbones,
ProteinMPNN produces sequences for them, and ColabFold folds those sequences so
they can be compared with the intended backbone.

`structbio` validates the sequences, warns when they would leave the machine,
and translates the configuration into `colabfold_batch` arguments. It does not
install ColabFold, its weights, or its parameter cache.

## Before the first run

ColabFold normally puts `colabfold_batch` on PATH, so no path is needed:

```yaml
tools:
  colabfold:
    executable: colabfold_batch
    manager: none
```

If it lives in a Conda environment instead:

```yaml
tools:
  colabfold:
    executable: colabfold_batch
    manager: conda
    environment: colabfold
```

Check it with `structbio doctor`. The first real run downloads AlphaFold
parameters into ColabFold's own cache, which is slow but happens only once.

## Quick command

```bash
colabfold predict my_sequences my_folds -n 5 --relax 1
```

The arguments are the sequences and the output folder.

| Option | Meaning |
| --- | --- |
| `-n, --num-models 1..5` | Models per sequence. Default 5. |
| `--msa-mode` | `mmseqs2_uniref_env` (default), `mmseqs2_uniref_env_envpair`, `mmseqs2_uniref`, or `single_sequence`. |
| `--templates` | Use PDB templates. |
| `--relax N` | Amber-relax the top N ranked models, on the GPU. |
| `--recycle N` | Recycle iterations; more is slower and sometimes better. |
| `--gpu 0`, `--gpu auto` | Card selection. |
| `--dry-run`, `--quiet` | Preview the command; suppress live output. |

## Sequences leave the machine by default

This is the one thing to understand before using ColabFold on unpublished work.
Every `mmseqs2_*` MSA mode builds the alignment by sending your sequences to the
**public ColabFold MMseqs2 server**. `structbio` prints a warning whenever a
configuration would do that:

```text
WARNING: MSAs will be built by the public ColabFold MMseqs2 server, so every
input sequence leaves this machine. Use msa.mode: single_sequence, or set
msa.host_url to a server you run, to keep sequences local
```

Two ways to keep sequences local:

```bash
colabfold predict my_sequences my_folds --msa-mode single_sequence
```

```yaml
msa:
  host_url: http://msa-server.inside.the.lab:8888
```

For designed sequences `single_sequence` is usually the right choice anyway: a
de novo design has no natural homologues, so an MSA contributes little.

## Folding a ProteinMPNN run

Point ColabFold at a ProteinMPNN output folder and it finds the designed
sequences by itself, because ProteinMPNN writes them to `OUTPUT/seqs/NAME.fa`:

```bash
rfdiffusion monomer 150 my_backbones -n 20
proteinmpnn design my_backbones 4 my_sequences
colabfold predict my_sequences my_folds --msa-mode single_sequence
```

The validation report says which folder it chose:

```text
Using designed sequences from /path/to/my_sequences/seqs
```

ProteinMPNN lays its output out two different ways, and both are handled:

| ProteinMPNN input | Layout | What ColabFold is given |
| --- | --- | --- |
| One PDB | `OUTPUT/seqs/*.fa` | that `seqs` folder |
| A folder of PDBs | `OUTPUT/STRUCTURE/seqs/*.fa` | one merged FASTA |

For a batch, the per-structure files are concatenated verbatim into
`OUTPUT/.structbio/inputs/sequences.fa` and ColabFold is run once over the lot.
`colabfold_batch` takes a single input path, and loading its models once for the
whole batch is much faster than once per design folder. No sequence is renamed,
reordered, or rewrapped, and the original ProteinMPNN files are not touched.

A folder that already contains its own FASTA files is used as given; the `seqs`
folders are only looked for when the folder holds no sequences of its own.

## What is checked before ColabFold starts

- every FASTA record parses, is named, and is non-empty;
- sequences contain only amino-acid characters, so a stray line of PDB text or
  a numeric column fails immediately rather than after the MSA stage;
- `:` separates chains of a complex, and empty chains are rejected;
- duplicate record names are reported, because they share one ColabFold job
  name and overwrite each other's results;
- `relax.num_relax` does not exceed `prediction.num_models`;
- a long sequence produces a memory warning.

CSV and a3m inputs are passed through to ColabFold, which validates them
itself; `structbio` says so rather than pretending to check them.

## Main YAML fields

| Field | Meaning |
| --- | --- |
| `input.sequences` | FASTA, CSV/TSV, a3m, a folder of them, or a ProteinMPNN output folder. |
| `msa.mode` | MSA source; see the warning above. |
| `msa.pair_mode` | `unpaired`, `paired`, or `unpaired_paired` for complexes. |
| `msa.templates` | Use PDB templates. |
| `msa.custom_template_path` | Directory of custom templates; requires `templates: true`. |
| `msa.host_url` | Your own MMseqs2 API server. |
| `msa.only` | Build MSAs and stop, for splitting CPU and GPU stages. |
| `prediction.num_models` | 1 to 5 models per sequence. |
| `prediction.num_recycle` | Recycle iterations. |
| `prediction.num_seeds`, `prediction.random_seed` | Sampling control. |
| `prediction.model_type` | `auto` picks monomer or multimer from the input. |
| `prediction.stop_at_score` | Stop early once a model scores this well. |
| `prediction.use_dropout` | Sample with dropout for diversity. |
| `relax.num_relax` | Amber-relax the top N ranked models. |
| `relax.use_gpu` | Relax on the GPU; requires `num_relax`. |
| `output.rank` | `auto`, `plddt`, `ptm`, `iptm`, or `multimer`. |
| `output.save_all`, `output.zip_results` | Extra outputs. |

`--overwrite-existing-results` is deliberately never generated. `structbio`
only ever runs into a new or empty output folder, so there is nothing to
overwrite; keeping the flag unavailable means a mistyped output name cannot
destroy a previous prediction.

## Outputs

ColabFold writes ranked PDB files, PAE and pLDDT plots, and JSON scores into
the output folder, named after each FASTA record. The ranked first model of
each sequence is the one to compare against the design target; `rank` decides
what "first" means.

If a fold fails after validation passed, compare `command.txt` with the
installed ColabFold's own `colabfold_batch --help`, and check that the
parameter download completed.
