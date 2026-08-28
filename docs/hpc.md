# HPC and SLURM

Cluster policy belongs in a named profile, not in Python code:

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
```

Select it from an experiment:

```yaml
resources:
  cluster: lab_gpu
  time: 08:00:00
```

Preview with `structbio TOOL submit CONFIG --dry-run`. Running the same command
without `--dry-run` creates an experiment and `job.slurm`, but still does not
submit. Add `--execute` to explicitly call `sbatch`. SLURM allocates GPUs; users
normally should not set `CUDA_VISIBLE_DEVICES` themselves.

The `resources.array` field accepts a SLURM array expression such as `0-99%10`.
Only use arrays when the backend/config maps the task index to genuinely
independent work.
