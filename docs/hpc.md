# HPC and SLURM

`structbio` can run in an existing local shell/allocation or generate a portable
SLURM script. It never submits a job merely because `submit` appears in the
command: `--execute` is required to call `sbatch`.

## Define a cluster profile

Cluster policy belongs in a named profile in the lab or user configuration, not
in Python code:

```yaml
cluster_profiles:
  lab_gpu:
    partition: gpu
    account: my_lab
    nodes: 1
    gpus: 1
    cpus: 16
    memory: 64G
    time: 04:00:00
    modules: [cuda]
    preamble:
      - export TMPDIR=/scratch/$USER/tmp
```

Choose names, modules, and preamble commands that match the actual cluster. A
profile can include:

| Field | Meaning |
| --- | --- |
| `partition` | SLURM partition. |
| `account` | Optional allocation/account. |
| `qos` | Optional quality of service. |
| `nodes` | Requested nodes. |
| `gpus` | Requested GPUs using `#SBATCH --gpus`. |
| `cpus` | CPUs per task. |
| `memory` | SLURM memory string such as `64G`. |
| `time` | Wall time in `HH:MM:SS` or `D-HH:MM:SS`. |
| `modules` | Modules loaded before the wrapped command. |
| `preamble` | Cluster-specific shell lines placed before the command. |

## Select resources in an experiment

Reference the profile and override job-specific values in the experiment YAML:

```yaml
resources:
  cluster: lab_gpu
  gpus: 1
  cpus: 16
  memory: 96G
  time: 08:00:00
```

More specific experiment values override profile values. Cluster profiles do not
hardcode a site into `structbio`; labs can define workstation, institutional,
and cloud profiles independently.

## Preview, prepare, and submit

First print a proposed script without writing anything:

```bash
structbio rfdiffusion submit design.yaml --dry-run
```

Check all `#SBATCH` lines, module loads, working directory, environment wrapper,
scientific command, and output paths. To create a recorded experiment plus
`job.slurm` without submitting it:

```bash
structbio rfdiffusion submit design.yaml
```

That command intentionally stops after writing the script. Inspect it at the
path printed by the CLI. To create a new experiment and submit it immediately:

```bash
structbio rfdiffusion submit design.yaml --execute
```

Each non-dry-run invocation creates a new experiment. The last command does not
submit a script prepared by the preceding command; it safely creates and submits
another uniquely recorded experiment.

SLURM assigns GPU visibility for the job. Users normally should not set
`CUDA_VISIBLE_DEVICES` manually. CryoZeta's verified upstream script performs its
own local GPU selection; see its wrapper guide for that special case.

## Monitor experiments and jobs

```bash
structbio status
structbio status tetra600_2026-08-28_001
```

Submitted experiment metadata stores the SLURM job ID and raw `sbatch` response.
When `squeue` is available, `status` displays the live scheduler state alongside
the recorded `structbio` state. Scheduler accounting after a job leaves `squeue`
is cluster-specific and is not queried automatically.

Inspect these files when a job fails:

```text
EXPERIMENT/job.slurm
EXPERIMENT/command.txt
EXPERIMENT/stdout.log
EXPERIMENT/stderr.log
EXPERIMENT/metadata.json
EXPERIMENT/environment.txt
```

## SLURM arrays

The `resources.array` field accepts a conservative SLURM array expression such
as `0-19%4`:

```yaml
resources:
  cluster: lab_gpu
  array: 0-19%4
  gpus: 1
```

Arrays are allowed only when the backend produces at least two independent
command steps. At present, the main use is a ProteinMPNN `input.directory`
batch: PDB files are sorted by filename, command step 0 handles the first file,
step 1 the second, and so on. The generated script uses
`SLURM_ARRAY_TASK_ID` to select exactly one command.

Before submission:

1. Count the PDB files accepted by the configuration.
2. Review `proteinmpnn inspect-mask` for every file.
3. Preview the generated SLURM script.
4. Set the array range to `0-(N-1)` for N generated steps.

If an array index has no matching step, the job exits with status 2 instead of
duplicating work. A one-command RFdiffusion or CryoZeta plan rejects an array
request because repeating that command could overwrite outputs inside the job.

## Cluster checklist

- Tool checkouts, model weights, raw inputs, and output roots must be visible
  from compute nodes.
- Conda/Pixi commands used by the installation must work in non-interactive
  batch shells.
- Profile modules and preamble lines must match the cluster.
- Requested CUDA and GPU types must be compatible with the wrapped tool.
- Use `--dry-run` after every profile or scheduler-policy change.
