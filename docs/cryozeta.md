# CryoZeta wrapper

The CryoZeta backend adapts the Kihara Lab inference scripts, verified against
`kiharalab/CryoZeta` main on 2026-08-28:

| Script | Used for | Flags structbio generates |
| --- | --- | --- |
| `inference_demo.sh` | Standard pipeline | `--input-json`, `--output-dir`, `--mode`, `--env`, `--gpu`, `--checkpoint`, `--interp-checkpoint` |
| `large_inference_demo.sh` | Complexes above ~2800 residues | `--input-json`, `--output-dir`, `--registration`, `--env`, `--gpu`, `--checkpoint`, `--detection-checkpoint` |

CryoZeta's input is a JSON file whose schema belongs to CryoZeta. `structbio`
can write that JSON for you from a map and a FASTA, which covers the common
case, and will also run a JSON you wrote yourself for anything more involved.

## Before the first run

CryoZeta is Linux/GPU software and uses Pixi in the verified upstream checkout.
Configure it as follows:

```yaml
tools:
  cryozeta:
    path: ~/software/CryoZeta
    executable: inference_demo.sh
    manager: pixi
    environment: default
```

Name `inference_demo.sh` even if you intend to run large complexes; the large
pipeline is its sibling in the same checkout and is selected with `--large`.

The checkout must already contain its Pixi environments, model assets,
detection checkpoint, inference checkpoints, and compiled dependencies. Follow
the installed CryoZeta version's setup instructions, then run:

```bash
structbio doctor
```

The standard upstream script runs inference stages on one GPU. Requesting more
than one GPU produces a warning; it does not make the wrapped standard pipeline
multi-GPU.

## The short way: a map and a FASTA

```bash
cryozeta predict emd_44046.map.gz chains.fasta my_model --resolution 2.99 --contour 0.3
```

The arguments are the density map, a FASTA holding every chain in the complex,
and the output folder. `--resolution` and `--contour` are required because
CryoZeta requires them and no sensible default exists: use the resolution and
the recommended contour level published with the map.

The native target JSON is written to `my_model/.structbio/inputs/targets.json`,
so you can read exactly what CryoZeta was asked to do.

| Option | Meaning |
| --- | --- |
| `--resolution`, `--contour` | Required map metadata. |
| `--mode` | `combined` (default), `cryozeta`, or `cryozeta-interpolate`. |
| `--large` | Use the large-complex pipeline. |
| `--registration` | Large-complex registration: `auto`, `teaser`, `svd`, `vesper`. |
| `--dna`, `--rna`, `--protein` | Name the FASTA records of a given type. |
| `--msa-dir`, `--pairing-db` | Precomputed MSA directory and pairing database. |
| `--gpu 0`, `--gpu auto` | Card selection, passed to CryoZeta's own `--gpu`. |

### Copies of a chain

Identical sequences are collapsed into one entry with a `count`, which is how
CryoZeta expects the copies of a homo-oligomer to be given. A FASTA holding the
same protein twice and one nucleic chain becomes:

```json
"sequences": [
  {"proteinChain": {"sequence": "MKTAYIAKQ", "count": 2}},
  {"dnaSequence": {"sequence": "ACGTACGT", "count": 1}}
]
```

### Chain types are never guessed

A, C, G and T are all valid amino acids as well as nucleotides, so a sequence
written only from those letters could be a short peptide or a DNA strand.
`structbio` refuses to choose:

```text
ERROR: Cannot tell whether 'chain_C' is protein, DNA, or RNA: it uses only the
letters A, C, G and T, which are valid in both. Name it in input.chains.dna,
input.chains.rna, or input.chains.protein
```

Resolve it with `--dna chain_C`, `--rna chain_C`, or `--protein chain_C`. A
sequence containing any other letter is protein, and one containing U but no T
is RNA; neither needs declaring.

## What is checked before CryoZeta starts

- the density map really is MRC/CCP4: the `MAP ` stamp is read at byte 208, the
  grid dimensions are reported, and `.gz` files are decompressed to check. A
  truncated download or a mistyped path fails in a second rather than after the
  detection stage;
- every FASTA record parses and holds only residue letters;
- checkpoint and precomputed-MSA paths exist;
- the total modelled residue count is reported, and a complex above roughly
  2800 residues is steered to `--large` — the standard scripts do not check
  this themselves and the README directs larger complexes to the other
  pipeline;
- checkpoints belonging to the other pipeline are rejected:
  `--interp-checkpoint` is standard-only and `--detection-checkpoint` is
  large-only.

`--overwrite` is deliberately never generated. `structbio` only ever runs into
a new or empty output folder, so a mistyped output name cannot re-run over an
earlier model.

## The long way: a hand-written target JSON

Ligands, ions, chain modifications, glycans, and several targets in one run
cannot be expressed by the short command. Write CryoZeta's own JSON and run it:

```bash
cryozeta predict-json targets.json my_models
```

The top level is a non-empty list. Each target needs `name`, `modelSeeds`,
`map_path`, `resolution`, `contour_level`, and `sequences`. Each entry of
`sequences` holds exactly one of `proteinChain`, `dnaSequence`, `rnaSequence`,
`ligand`, or `ion`; a polymer entry holds `sequence` and `count`, and may hold
`msa`, `modifications`, and — for proteins — `glycans`.

## The YAML form

Everything the short commands do is also expressible in YAML, either pointing at
a native JSON:

```yaml
tool: cryozeta

experiment:
  name: my_target

input:
  json: ~/project/config/my_target.json

mode: combined
pixi_environment: default

resources:
  gpus: 1
```

or describing the map and chains and letting `structbio` build the JSON:

```yaml
tool: cryozeta

experiment:
  name: my_target

input:
  map: ~/data/emd_44046.map.gz
  sequences: ~/data/chains.fasta
  resolution: 2.99
  contour_level: 0.3
  chains:
    dna: [chain_C]
  msa:
    precomputed_msa_dir: ~/data/msas
    pairing_db: uniref100

mode: combined

resources:
  gpus: 1
```

The supported modes map directly to the verified upstream script:

| Mode | Behavior |
| --- | --- |
| `combined` | Run detection, standard inference, interpolated inference, and combine the selected results. |
| `cryozeta` | Run detection and standard inference. |
| `cryozeta-interpolate` | Run detection and interpolated inference. |

Modes apply to the standard pipeline only. `large_inference_demo.sh` has no
`--mode`; it takes `--registration` instead, which `structbio` supplies when
`large: true`.

## Review the exact command

```bash
structbio cryozeta validate cryozeta.yaml
structbio cryozeta command cryozeta.yaml
structbio cryozeta run cryozeta.yaml --dry-run
```

Validation reads the map header, the sequences, and the JSON, but never loads a
model, compiles CUDA code, or runs inference.

On a shared cluster, `structbio cryozeta submit` writes a SLURM script instead;
see [the optional cluster guide](cluster.md).

## Main YAML fields

| Field | Meaning |
| --- | --- |
| `input.json` | CryoZeta native input JSON. Mutually exclusive with `input.map`. |
| `input.map` | Density map, with `input.sequences`, `input.resolution`, and `input.contour_level`. |
| `input.sequences` | FASTA of every chain in the complex. |
| `input.resolution` | Map resolution in angstroms. |
| `input.contour_level` | Recommended contour level for the map. |
| `input.target_name` | Name of the generated target; defaults to the experiment name. |
| `input.chains` | `protein`, `dna`, `rna` lists naming records whose type is ambiguous. |
| `input.msa` | `precomputed_msa_dir` and `pairing_db`, attached to protein and RNA chains. |
| `mode` | `combined`, `cryozeta`, or `cryozeta-interpolate`; standard pipeline only. |
| `large` | Use `large_inference_demo.sh` instead. |
| `registration` | `auto`, `teaser`, `svd`, or `vesper`; large pipeline only. |
| `detection_checkpoint` | Detection checkpoint override; large pipeline only. |
| `pixi_environment` | Optional upstream Pixi environment such as `default`, `cu11`, or `cu13`. |
| `gpu_ids` | Optional explicit local GPU indices passed to the upstream script. |
| `checkpoint` | Optional standard-model checkpoint override. |
| `interpolation_checkpoint` | Optional interpolated-model checkpoint override. |
| `resources` | Resource request recorded with the experiment, and used only by `submit`. |

`input.json` and `input.map` are mutually exclusive, and a configuration that
sets neither is rejected before anything runs.

## Checking the installation before a run

```bash
structbio env verify cryozeta
```

This runs code inside CryoZeta's own pixi environment and imports the modules
it needs, so an incomplete installation is found in seconds rather than after
the detection stage.

The one worth knowing about is TEASER++:

```text
FAILED: teaserpp_python could not be imported, so CryoZeta cannot load its
        fitting module.
```

`pixi run setup` builds TEASER++ through a `build-teaser` task whose command is
conditional on `externals/TEASER-plusplus/build/libteaser.so` existing. A build
that stopped part-way leaves that file behind, so a later `pixi run setup`
skips the step and reports success while the Python bindings are still absent.
Every mode fails at import, because `cryozeta/model/modules/fitting.py` imports
it unconditionally.

TEASER++ is compiled from source and is not published as a wheel, so it has to
build successfully here. It needs CMake, a C++ compiler, Eigen and Boost. Put
them in the environment the build runs in:

```bash
cd ~/software/CryoZeta
pixi add cmake eigen boost-cpp cxx-compiler
pixi run build-teaser
```

The first line is what `cmake: command not found` is asking for. Adding the
tools to the pixi environment is more reliable than installing them elsewhere
and hoping the task finds them on PATH, and it keeps the compiler and the
libraries consistent with the environment CryoZeta itself runs in. It edits
`pyproject.toml` and `pixi.lock`, which are tracked by git, so
`git checkout pyproject.toml pixi.lock` reverts it.

If `build-teaser` reports nothing to do, an earlier attempt left
`externals/TEASER-plusplus/build/libteaser.so` behind and the task skips
itself. Remove that directory and run it again:

```bash
rm -rf externals/TEASER-plusplus/build
pixi run build-teaser
```

Then confirm:

```bash
structbio env verify cryozeta
```

## Outputs and limitations

After a run, structbio lists every model file it finds and the chains in each,
against the number of chains the request described:

```text
Model files (2):
  combined/final_model.pdb        2 chains (A:40, B:40)
  detection/detected_atoms.pdb   16 chains (A:3, B:3, C:3, ...)

Chains requested: 2
```

**Read that before judging a prediction.** In `combined` mode CryoZeta writes
detection output and intermediates alongside the final model, and the detection
output is exactly what a broken model looks like: many short fragments in many
chains. It is not the answer; it is the raw atom detection the answer is built
from. The file whose chain count matches the request is the one to open.

The upstream pipeline writes into the output folder a quick command names, or
below `EXPERIMENT/outputs/` for a YAML run. In combined mode, the important
upstream directories normally include detection output, standard and
interpolated predictions, and final combined selections. Consult the installed
CryoZeta documentation for exact filenames and ranking metrics.

Not wrapped, and needing a hand-written JSON: ligands, ions, chain
modifications, and glycans. Not wrapped at all: the large pipeline's
`--example` selector, which picks from CryoZeta's own bundled examples;
`structbio` always supplies an input JSON instead.

If validation passes but execution fails, inspect `stderr.log` for upstream
problems involving assets, MSA contents, Pixi, CUDA, GPU memory, compiler
caches, or the installed CryoZeta version, and compare `command.txt` with the
installed script's own `--help`.
