# Example configurations

Copy an example into a project working directory and edit the copy. Do not run a
template containing `/absolute/path/...` until every placeholder has been
replaced.

## RFdiffusion

- `rfdiffusion/monomer.yaml`: unconditional 150-residue monomers; no input PDB.
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
- `cryozeta/dataset.yaml`: wrapper YAML pointing at a native JSON file.

The JSON schema is owned by CryoZeta. Compare the template with the installed
CryoZeta version before use.
