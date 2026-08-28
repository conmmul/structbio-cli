# Repository map

- `src/structbio/`: CLI, configuration, experiment tracking, validation, and SLURM.
- `src/structbio/tools/`: isolated scientific-tool backends.
- `tests/`: GPU-free unit and CLI tests with small PDB fixtures.
- `examples/`: researcher-facing YAML examples.
- `docs/`: architecture and tool-specific behavior.

Preserve scientific correctness over convenience. Never guess third-party CLI
syntax: check the installed version and its official documentation. Preserve
residue numbering and chain identifiers, avoid destructive operations, use dry
runs wherever possible, and never weaken an overwrite guard. Add tests for every
backend change. Maintain compatibility with existing YAML files and update the
relevant documentation whenever behavior changes.
