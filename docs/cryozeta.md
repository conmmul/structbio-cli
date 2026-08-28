# CryoZeta

This is intentionally a thin adapter to the Kihara Lab CryoZeta
`inference_demo.sh` interface verified from the official repository. The YAML
points to CryoZeta's native input JSON rather than duplicating that evolving
schema.

```bash
structbio cryozeta validate examples/cryozeta/dataset.yaml
structbio cryozeta command examples/cryozeta/dataset.yaml
```

The JSON must be a non-empty list whose entries contain `name`, `modelSeeds`,
`map_path`, `resolution`, `contour_level`, and `sequences`. Referenced maps must
exist. The adapter never passes CryoZeta's `--overwrite` option.

CryoZeta is Linux/GPU software and currently uses Pixi upstream. Confirm the
installed checkout matches the documented script before execution. Large-complex
mode is not wrapped yet.
