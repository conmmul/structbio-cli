# Architecture and adding backends

Core code owns configuration precedence, experiment directories, metadata,
execution, diagnostics, and SLURM. A backend owns only tool-specific models,
validation, command construction, generated sidecar files, and output discovery.

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
written below a newly allocated experiment directory.
