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

## ColabFold

- `colabfold/designs.yaml`: fold designed sequences without contacting the
  public MSA server. The same run in short form:
  `colabfold predict my_sequences my_folds --msa-mode single_sequence`.
- `colabfold/natural_msa.yaml`: a natural sequence with a full MSA and
  templates. This mode sends sequences to a public server.
- `colabfold/complex.yaml`: a complex, whose chains are joined by `:` inside one
  FASTA record.

## CryoZeta

- `cryozeta/map_and_sequences.yaml`: a map plus a FASTA, with the native target
  JSON generated for you. The same run in short form:
  `cryozeta predict map.map.gz chains.fasta my_model --resolution 2.99 --contour 0.3`.
- `cryozeta/large_complex.yaml`: the large-complex pipeline, for complexes above
  roughly 2800 residues.
- `cryozeta/native_input.example.json`: schematic native CryoZeta target JSON,
  for ligands, ions, modifications, or several targets at once.
- `cryozeta/dataset.yaml`: wrapper YAML pointing at a native JSON file. The same
  run in short form: `cryozeta predict-json targets.json my_models`.

The JSON schema is owned by CryoZeta. Compare the template with the installed
CryoZeta version before use.

## Workstation configuration

- `lab-config.yaml`: tool installation paths for a whole workstation, used
  through `STRUCTBIO_LAB_CONFIG` or copied to
  `~/.config/structbio/config.yaml`.
