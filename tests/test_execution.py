import sys
from pathlib import Path

from structbio.execution import execute_plan
from structbio.experiment import prepare_output_dir, write_records
from structbio.tools.base import CommandPlan, CommandStep


def _plan(tmp_path: Path, body: str) -> CommandPlan:
    script = tmp_path / "tool.py"
    script.write_text(body)
    return CommandPlan(
        steps=[CommandStep(argv=(sys.executable, str(script)))],
        output_dir=tmp_path / "out",
    )


def _prepared(tmp_path: Path) -> object:
    paths = prepare_output_dir(tmp_path / "out")
    write_records(
        paths,
        config={"tool": "test"},
        command="python tool.py",
        tool_name="test",
        tool_path=None,
        input_paths=[],
        status="prepared",
    )
    return paths


def test_output_is_streamed_and_logged(tmp_path: Path, capfd) -> None:
    paths = _prepared(tmp_path)
    plan = _plan(
        tmp_path,
        "import sys\nprint('progress line')\nprint('a warning', file=sys.stderr)\n",
    )
    assert execute_plan(plan, paths) == 0
    captured = capfd.readouterr()
    assert "progress line" in captured.out
    assert "a warning" in captured.err
    assert "progress line" in paths.stdout.read_text()
    assert "a warning" in paths.stderr.read_text()


def test_quiet_still_logs_everything(tmp_path: Path, capfd) -> None:
    paths = _prepared(tmp_path)
    plan = _plan(tmp_path, "print('progress line')\n")
    assert execute_plan(plan, paths, stream=False) == 0
    assert "progress line" not in capfd.readouterr().out
    assert "progress line" in paths.stdout.read_text()


def test_wrapped_python_tools_are_unbuffered(tmp_path: Path, capfd) -> None:
    """Without PYTHONUNBUFFERED a piped child holds its output until it exits."""

    paths = _prepared(tmp_path)
    plan = _plan(tmp_path, "import os\nprint(os.environ.get('PYTHONUNBUFFERED'))\n")
    execute_plan(plan, paths)
    assert capfd.readouterr().out.strip().endswith("1")


def test_step_environment_overrides_the_default(tmp_path: Path, capfd) -> None:
    paths = _prepared(tmp_path)
    script = tmp_path / "tool.py"
    script.write_text("import os\nprint(os.environ.get('CUDA_VISIBLE_DEVICES'))\n")
    plan = CommandPlan(
        steps=[
            CommandStep(
                argv=(sys.executable, str(script)), env={"CUDA_VISIBLE_DEVICES": "2"}
            )
        ],
        output_dir=tmp_path / "out",
    )
    execute_plan(plan, paths)
    assert capfd.readouterr().out.strip().endswith("2")


def test_a_failing_step_stops_the_plan_and_is_recorded(tmp_path: Path) -> None:
    import json

    paths = _prepared(tmp_path)
    failing = tmp_path / "fail.py"
    failing.write_text("import sys\nsys.exit(3)\n")
    second = tmp_path / "second.py"
    second.write_text("print('should not run')\n")
    plan = CommandPlan(
        steps=[
            CommandStep(argv=(sys.executable, str(failing)), name="first"),
            CommandStep(argv=(sys.executable, str(second)), name="second"),
        ],
        output_dir=tmp_path / "out",
    )
    assert execute_plan(plan, paths, stream=False) == 3
    assert "should not run" not in paths.stdout.read_text()
    metadata = json.loads(paths.metadata.read_text())
    assert metadata["status"] == "failed"
    assert metadata["return_code"] == 3


def test_a_missing_program_is_reported_not_raised(tmp_path: Path) -> None:
    paths = _prepared(tmp_path)
    plan = CommandPlan(
        steps=[CommandStep(argv=(str(tmp_path / "not-a-program"),))],
        output_dir=tmp_path / "out",
    )
    assert execute_plan(plan, paths, stream=False) == 127
    assert "Unable to start" in paths.stderr.read_text()
