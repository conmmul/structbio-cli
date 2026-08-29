# Repository map

- `src/structbio/`: short positional commands, configuration, output and
  experiment tracking, validation, and the optional SLURM script generator.
- `src/structbio/cli/`: the Typer interface, one module per kind of command;
  `cli/__init__.py` imports them in the order `structbio --help` lists them.
- `src/structbio/tools/`: isolated scientific-tool backends.
- `tests/`: GPU-free unit and CLI tests with small PDB fixtures. The autouse
  `isolated_workstation` fixture gives every test its own home and
  configuration; nothing may read or write the real ones.
- `examples/`: researcher-facing YAML examples.
- `docs/`: architecture and tool-specific behavior.
- `install.sh`: the one command a new researcher runs; keep it working from a
  bare clone, and keep it safe to re-run.
- `PROTOCOL.md`: the beginner-facing guide; keep every command in it real, and
  check it still matches when an interface changes.
- `bin/`: generated shell wrappers, one per backend; keep them in step with
  `structbio.wrappers.render_wrapper`.

Run `ruff check .` and `pytest` before finishing.

A new backend needs an entry in `discovery.SIGNATURES` and `install.RECIPES`;
both are covered by tests that compare them against the backend registry.
Install recipes are quoted from the upstream README and carry the date it was
read. Never make structbio create environments, download weights, or accept a
licence on the researcher's behalf.

This is workstation software first. Short positional commands are the primary
interface, results belong in the folder the researcher named, and SLURM support
is an optional extra for shared clusters.

Prefer doing a step for the researcher over telling them to do it, as long as
it is reversible and announced: finding an installed tool, recording a path,
appending a PATH line, adopting an environment that already passes its check.
Everything that is not — building an environment, downloading weights,
accepting a licence, overwriting results — stays an explicit command they type.

Preserve scientific correctness over convenience. Never guess third-party CLI
syntax: check the installed version and its official documentation, and record
the verified version in the backend docstring. Warn the researcher when a
wrapped tool sends their data off the machine. Preserve
residue numbering and chain identifiers, avoid destructive operations, use dry
runs wherever possible, and never weaken an overwrite guard. Add tests for every
backend change. Maintain compatibility with existing YAML files and update the
relevant documentation whenever behavior changes.
