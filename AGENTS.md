# Repository map

- `src/structbio/`: CLI, short positional commands, configuration, output and
  experiment tracking, validation, and the optional SLURM script generator.
- `src/structbio/tools/`: isolated scientific-tool backends.
- `tests/`: GPU-free unit and CLI tests with small PDB fixtures.
- `examples/`: researcher-facing YAML examples.
- `docs/`: architecture and tool-specific behavior.
- `bin/`: generated shell wrappers, one per backend; keep them in step with
  `structbio.wrappers.render_wrapper`.

This is workstation software first. Short positional commands are the primary
interface, results belong in the folder the researcher named, and SLURM support
is an optional extra for shared clusters.

Preserve scientific correctness over convenience. Never guess third-party CLI
syntax: check the installed version and its official documentation. Preserve
residue numbering and chain identifiers, avoid destructive operations, use dry
runs wherever possible, and never weaken an overwrite guard. Add tests for every
backend change. Maintain compatibility with existing YAML files and update the
relevant documentation whenever behavior changes.
