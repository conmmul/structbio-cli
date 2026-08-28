# RFdiffusion

The backend targets the official `scripts/run_inference.py` Hydra interface. It
supports monomers, symmetry (`cN`, `dN`, or `tetrahedral`), motif scaffolding,
binders, partial diffusion, sequence/structure inpainting, and named guiding
potentials.

```bash
structbio rfdiffusion validate examples/rfdiffusion/tetrahedral.yaml
structbio rfdiffusion command examples/rfdiffusion/tetrahedral.yaml
structbio rfdiffusion run examples/rfdiffusion/tetrahedral.yaml --dry-run
```

For input-conditioned work, contig PDB references, hotspot residues, and
inpainting selections are checked against the original chain IDs and residue
numbers. `hydra_overrides` is available for documented upstream settings; keys
are syntax-checked but their scientific meaning remains the researcher's
responsibility.

Symmetric total lengths must obey RFdiffusion's upstream symmetry constraints.
Start guiding-potential studies with an unguided baseline, as recommended by the
upstream documentation.
