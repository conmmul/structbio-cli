# Architecture and adding backends

Core code owns configuration precedence, output and experiment directories,
metadata, execution, diagnostics, the short positional command grammar, and the
optional SLURM script generator. A backend owns only tool-specific models,
validation, command construction, generated sidecar files, and output discovery.

| Module | Responsibility |
| --- | --- |
| `cli.py` | Typer commands: the short positional form and the YAML form. |
| `quick.py` | Positional arguments to configuration mappings, plus contigs derived from a structure. |
| `config.py` | Layered configuration, from package defaults to `--set`. |
| `experiment.py` | Output folders, dated experiment folders, provenance records. |
| `execution.py` | Sequential, no-shell execution of a command plan. |
| `wrappers.py` | The generated per-tool shell commands. |
| `slurm.py` | Optional cluster script generation. |

Implement `ToolBackend` in `src/structbio/tools/`:

```python
class NewBackend(ToolBackend):
    name = "newtool"
    display_name = "NewTool"

    def parse_config(self, raw, source): ...
    def validate(self, config): ...
    def build_command(self, config, context): ...
    def check_environment(self, installation): ...
    def collect_outputs(self, experiment_dir): ...
```

Return argument vectors in `CommandStep`; never construct a shell command for
execution. Register the backend in `tools/__init__.py`, add its Typer group, test
validation and exact command generation without the external program, and
document the verified upstream version/interface. Generated files may only be
written below a newly created output or experiment directory.

A new backend should also offer at least one short positional command, built
from a fragment function in `quick.py`, for whatever run type researchers
perform most often; anything that cannot be said in three or four arguments
stays in YAML. Registering the backend automatically gives it a generated shell
command, so re-run `structbio install-wrappers` and refresh `bin/` afterwards.
