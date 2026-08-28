# RFdiffusion wrapper

The RFdiffusion backend targets the official `scripts/run_inference.py` Hydra
interface. `structbio` validates the researcher-facing YAML, checks referenced
PDB chains and residue numbers, and translates it into Hydra arguments. It does
not run sequence design; RFdiffusion outputs designed backbones.

## Before the first run

Confirm the installation in the lab or user configuration:

```yaml
tools:
  rfdiffusion:
    path: /work/software/RFdiffusion
    executable: scripts/run_inference.py
    manager: conda
    environment: SE3nv
```

Then check the host:

```bash
structbio doctor
structbio rfdiffusion command examples/rfdiffusion/tetrahedral.yaml
```

The RFdiffusion checkout, environment, model weights, and CUDA-compatible PyTorch
must already be installed according to the upstream project.

## Recommended command sequence

```bash
structbio rfdiffusion validate design.yaml
structbio rfdiffusion command design.yaml
structbio rfdiffusion run design.yaml --dry-run
```

After reviewing all three outputs, choose one execution path:

```bash
# Run in the current shell and wait for completion.
structbio rfdiffusion run design.yaml

# Or preview and explicitly submit a SLURM job.
structbio rfdiffusion submit design.yaml --dry-run
structbio rfdiffusion submit design.yaml --execute
```

Each invocation without `--dry-run` creates a new experiment. Running
`submit design.yaml` without `--execute` prepares `job.slurm` but does not submit
it. A later `submit design.yaml --execute` creates another experiment; it does
not reuse the previously prepared directory.

## Minimal unconditional monomer

```yaml
tool: rfdiffusion

experiment:
  name: monomer150

input:
  pdb: null

design:
  mode: monomer
  length: 150
  num_designs: 10

resources:
  gpus: 1
  cpus: 8
  memory: 32G
  time: 02:00:00
```

`length: 150` becomes an exact `150-150` RFdiffusion contig. The output prefix
is always placed under the new experiment's `outputs/` directory.

## Symmetric oligomer generation

Use `mode: symmetry` and one supported symmetry value: `cN`, `dN`, or
`tetrahedral`.

```yaml
tool: rfdiffusion

experiment:
  name: tetra600

design:
  mode: symmetry
  symmetry: tetrahedral
  length: 600
  num_designs: 20

potentials:
  olig_contacts:
    weight_intra: 1.0
    weight_inter: 0.05

diffusion:
  guide_scale: 0.5
  guide_decay: quadratic

resources:
  gpus: 1
```

For symmetry mode, `length` is the total oligomer length. It must satisfy the
upstream RFdiffusion divisibility and symmetry constraints. `structbio` checks
the syntax and supported symmetry name but cannot determine whether a proposed
architecture is scientifically appropriate.

## Motif scaffolding

Input-conditioned modes require an existing PDB. Paths relative to the YAML are
resolved relative to the YAML file, not the current shell directory.

```yaml
tool: rfdiffusion

experiment:
  name: scaffold_active_site

input:
  pdb: ../inputs/enzyme_motif.pdb

design:
  mode: motif
  contigs:
    - 5-15/A10-25/30-40
  num_designs: 100

resources:
  gpus: 1
```

Here, `A10-25` refers to the original PDB chain and numbering. Validation fails
if chain A or any residue from 10 through 25 is missing. `structbio` does not
renumber the PDB.

## Binder design

```yaml
tool: rfdiffusion

experiment:
  name: target_binder

input:
  pdb: ../inputs/target.pdb

design:
  mode: binder
  contigs:
    - B1-100/0 100-100
  hotspot_residues: [B30, B33, B34]
  num_designs: 1000

resources:
  gpus: 1
```

RFdiffusion's `/0 ` chain-break notation includes a significant space; keep the
entire contig as one YAML scalar. Hotspot and inpaint lists use RFdiffusion's
upstream residue notation (`B30`, `B33`, `B34`). They are still validated
against the original PDB.

## Partial diffusion

```yaml
tool: rfdiffusion

experiment:
  name: diversify_complex

input:
  pdb: ../inputs/complex.pdb

design:
  mode: partial
  contigs:
    - 100-100/0 B1-150
  num_designs: 50

diffusion:
  timesteps: 50
  partial_t: 20
```

Partial mode requires `diffusion.partial_t`. If `timesteps` is supplied,
`partial_t` cannot be larger. RFdiffusion also requires the produced contig to
have the same total length as the input being partially diffused; review the
printed upstream command and upstream partial-diffusion guidance carefully.

## Sequence and structure inpainting

```yaml
tool: rfdiffusion

experiment:
  name: inpaint_target

input:
  pdb: ../inputs/target.pdb

design:
  mode: inpainting
  contigs:
    - 70-100/0 B165-178
  inpaint_sequence: [B165-170]
  inpaint_structure: [B171-178]
  num_designs: 20
```

Every listed chain and residue is checked. Sequence and structure inpainting
select different upstream RFdiffusion behaviors, so do not interchange them.

## Guiding potentials and advanced overrides

Named mappings under `potentials` become entries in
`potentials.guiding_potentials`:

```yaml
potentials:
  olig_contacts:
    weight_intra: 1.0
    weight_inter: 0.1

diffusion:
  guide_scale: 2.0
  guide_decay: quadratic
```

Use an unguided baseline before tuning potential weights. For upstream settings
that do not yet have a first-class YAML field, use `hydra_overrides` only after
checking the installed RFdiffusion version:

```yaml
hydra_overrides:
  denoiser.noise_scale_ca: 0.5
  denoiser.noise_scale_frame: 0.5
```

Override keys are syntax-checked, but `structbio` cannot validate their model
compatibility or scientific meaning.

## Main YAML fields

| Field | Meaning |
| --- | --- |
| `experiment.name` | Prefix for the unique experiment directory and output files. |
| `input.pdb` | Optional PDB; required for motif, binder, partial, and inpainting modes. |
| `design.mode` | `monomer`, `symmetry`, `motif`, `binder`, `partial`, or `inpainting`. |
| `design.length` | Exact generated length when explicit contigs are not used. |
| `design.contigs` | RFdiffusion contig strings; takes precedence over `length`. |
| `design.num_designs` | Number of RFdiffusion trajectories. |
| `design.symmetry` | `cN`, `dN`, or `tetrahedral` in symmetry mode. |
| `design.hotspot_residues` | Binder hotspot residues in upstream notation. |
| `design.inpaint_sequence` | Residues whose sequence identity is masked. |
| `design.inpaint_structure` | Residues whose structure is masked. |
| `diffusion.timesteps` | Optional `diffuser.T` override. |
| `diffusion.partial_t` | Required noising time for partial diffusion. |
| `potentials` | Named guiding potentials and their parameters. |
| `hydra_overrides` | Pass-through for documented installed-version settings. |

## Outputs and troubleshooting

RFdiffusion output is written below `EXPERIMENT/outputs/`. The experiment also
contains the exact Hydra command and logs. If validation fails:

- confirm the PDB path is visible on the current host and compute nodes;
- compare chain IDs and residue numbers with the original PDB;
- keep chain breaks and contigs as a single YAML string;
- use `command` to review quoting and translated Hydra keys;
- use `doctor` to check the configured checkout, Conda, CUDA, and GPU visibility.
