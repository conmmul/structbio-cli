# ProteinMPNN

ProteinMPNN internally represents fixed positions as one-based ordinal indices
within each chain. Researchers, however, specify selections using the PDB's
original numbering, for example `A:697-749`. `structbio` translates between the
two systems and then reconstructs the mutable set from the generated dictionary.
If it differs by even one residue—or is the complement—the run is aborted.

```bash
structbio proteinmpnn inspect-mask examples/proteinmpnn/design_region.yaml
structbio proteinmpnn run examples/proteinmpnn/design_region.yaml --dry-run
```

With no `designable_positions`, all residues in `design.chains` are designable
except `fixed_positions`. With `designable_positions`, only those exact residues
are designable. Other chains are fixed. The backend also supports global and
position-specific omitted amino acids, global and position-specific AA biases,
soluble weights, temperatures, multiple sequences, and a directory of PDBs.

Insertion codes are supported for individual selections such as `A:10A`; ranges
containing insertion codes are rejected rather than silently reinterpreted.
