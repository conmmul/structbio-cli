# Troubleshooting

Start with `structbio doctor`, then inspect the generated command using
`structbio TOOL command CONFIG.yaml`.

- **NOT CONFIGURED:** add the tool path, executable, manager, and environment to
  the lab or user config.
- **CONFIGURED, UNAVAILABLE:** the configured script or environment manager is
  missing on this host.
- **Missing chain/residue:** inspect the original PDB. `structbio` will not
  renumber residues or guess a chain.
- **Experiment already exists:** a new numeric suffix is selected automatically;
  existing records are never overwritten.
- **SLURM unavailable:** generate the script on a workstation or login node, then
  use a host with `sbatch` only when ready.
- **CryoZeta validation failure:** validate the native input JSON against the
  installed CryoZeta version and ensure map/MSA paths are visible on the compute
  node.
