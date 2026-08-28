# CryoZeta wrapper

The CryoZeta backend is intentionally a thin adapter to the Kihara Lab
`inference_demo.sh` interface. CryoZeta's native input schema is JSON and changes
with the upstream program, so the `structbio` YAML points to a native CryoZeta
JSON file rather than duplicating that schema.

Large-complex mode is not wrapped. This guide covers the verified standard
pipeline: atom detection, standard and/or interpolated inference, and optional
result combination.

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

The checkout must already contain its Pixi environments, model assets,
detection checkpoint, inference checkpoints, and compiled dependencies. Follow
the installed CryoZeta version's setup instructions, then run:

```bash
structbio doctor
```

The standard upstream script runs inference stages on one GPU. Requesting more
than one GPU produces a warning; it does not make the wrapped standard pipeline
multi-GPU.

## 1. Prepare CryoZeta's native input JSON

The top level must be a non-empty list. Each target entry must contain `name`,
`modelSeeds`, `map_path`, `resolution`, `contour_level`, and `sequences`:

```json
[
  {
    "name": "my_target",
    "modelSeeds": [],
    "map_path": "/work/project/maps/my_target.map.gz",
    "resolution": 3.2,
    "contour_level": 0.25,
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MKT...",
          "count": 1,
          "msa": {
            "precomputed_msa_dir": "/work/project/msa/chain_a",
            "pairing_db": "uniref100"
          }
        }
      }
    ]
  }
]
```

This example is schematic: replace `MKT...` with a valid full sequence and
follow the exact schema required by the installed CryoZeta version. Protein and
RNA inputs require the upstream MSA fields/files; DNA handling differs. The
wrapper verifies required top-level fields and the map path, but upstream
CryoZeta remains responsible for full sequence/MSA schema validation.

Use absolute map and MSA paths so their meaning does not depend on the directory
the command was typed in. Raw maps and MSA directories are never modified by
`structbio`.

## 2. Run it, or prepare a wrapper YAML

For the standard pipeline the quick command is enough:

```bash
cryozeta predict my_target.json my_maps --mode combined --gpu 0
```

The arguments are the native JSON and the output folder. `--gpu` is passed to the
upstream script as its own GPU selection rather than through
`CUDA_VISIBLE_DEVICES`. Checkpoint overrides and an explicit Pixi environment
need the YAML form below.



```yaml
tool: cryozeta

experiment:
  name: my_target_cryozeta

input:
  json: ~/project/config/my_target.json

mode: combined
pixi_environment: default

resources:
  gpus: 1
```

The supported modes map directly to the verified upstream script:

| Mode | Behavior |
| --- | --- |
| `combined` | Run detection, standard inference, interpolated inference, and combine the selected results. |
| `cryozeta` | Run detection and standard inference. |
| `cryozeta-interpolate` | Run detection and interpolated inference. |

## 3. Validate and review the exact command

```bash
structbio cryozeta validate cryozeta.yaml
structbio cryozeta command cryozeta.yaml
structbio cryozeta run cryozeta.yaml --dry-run
```

Validation checks that the native JSON parses, contains at least one target,
has the required top-level fields, and references an existing map. It does not
load a model, compile CUDA code, or run inference.

## 4. Run it

```bash
structbio cryozeta run cryozeta.yaml
```

On a shared cluster, `structbio cryozeta submit` writes a SLURM script instead;
see [the optional cluster guide](cluster.md).

If the upstream Pixi environment should be selected explicitly, use
`pixi_environment`. Optional local GPU IDs and checkpoint overrides are also
available:

```yaml
pixi_environment: cu11
gpu_ids: [0]
checkpoint: ~/software/CryoZeta/assets/custom.safetensors
interpolation_checkpoint: ~/software/CryoZeta/assets/custom-interpolate.safetensors
```

`gpu_ids` selects cards on a workstation; omit it under a scheduler and let the
allocation expose the assigned GPU. Only set checkpoint overrides when they are
compatible with the installed CryoZeta interface and model.

## Main YAML fields

| Field | Meaning |
| --- | --- |
| `input.json` | CryoZeta native input JSON. |
| `mode` | `combined`, `cryozeta`, or `cryozeta-interpolate`. |
| `pixi_environment` | Optional upstream Pixi environment such as `default`, `cu11`, or `cu13`. |
| `gpu_ids` | Optional explicit local GPU indices passed to the upstream script. |
| `checkpoint` | Optional standard-model checkpoint override. |
| `interpolation_checkpoint` | Optional interpolated-model checkpoint override. |
| `resources` | Resource request recorded with the experiment, and used only by `submit`. |

## Outputs and limitations

The upstream pipeline writes into the output folder a quick command names, or
below `EXPERIMENT/outputs/` for a YAML run. In combined mode, the
important upstream directories normally include detection output, standard and
interpolated predictions, and final combined selections. Consult the installed
CryoZeta documentation for exact filenames and ranking metrics.

The adapter never passes CryoZeta's `--overwrite` option. A new `structbio`
experiment is created instead. If validation passes but execution fails, inspect
`stderr.log` for upstream problems involving assets, MSA contents, Pixi, CUDA,
GPU memory, compiler caches, or the installed CryoZeta version.
