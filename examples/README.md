# Example configurations

These YAML files are for runs the short commands do not cover; for the common
cases, `rfdiffusion monomer 150 my_monomers` and its siblings need no file at
all. See the [README](../README.md#quick-commands) for the full list.

Copy an example into a project working directory and edit the copy. Do not run a
template containing `/absolute/path/...` until every placeholder has been
replaced.

## RFdiffusion

- `rfdiffusion/monomer.yaml`: unconditional 150-residue monomers; no input PDB.
  The same run in short form: `rfdiffusion monomer 150 my_monomers -n 10`.
- `rfdiffusion/tetrahedral.yaml`: tetrahedral symmetric oligomers with an
  oligomer-contact guiding potential.
- `rfdiffusion/binder.yaml`: binder design against chain B with hotspot residues.
- `rfdiffusion/partial.yaml`: partial diffusion of an existing complex.

Review the translated Hydra command with:

```bash
structbio rfdiffusion command CONFIG.yaml
```

## ProteinMPNN

- `proteinmpnn/all_chains.yaml`: design every parsed chain in one PDB.
- `proteinmpnn/design_region.yaml`: design only one original PDB residue range.
- `proteinmpnn/batch_directory.yaml`: run the same chain design across a PDB
  directory.

Always inspect the mask before running:

```bash
structbio proteinmpnn inspect-mask CONFIG.yaml
```

## CryoZeta

- `cryozeta/native_input.example.json`: schematic native CryoZeta target JSON.
- `cryozeta/dataset.yaml`: wrapper YAML pointing at a native JSON file. The same
  run in short form: `cryozeta predict targets.json my_maps`.

The JSON schema is owned by CryoZeta. Compare the template with the installed
CryoZeta version before use.

## Workstation configuration

- `lab-config.yaml`: tool installation paths for a whole workstation, used
  through `STRUCTBIO_LAB_CONFIG` or copied to
  `~/.config/structbio/config.yaml`.
