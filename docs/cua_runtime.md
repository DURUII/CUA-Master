# Canonical CUA Runtime

`cuamaster.runtime` is the reusable execution layer for canonical GUI, CLI,
tool, and termination actions. It is intentionally benchmark-independent.

Install it from this checkout:

```bash
pip install -e .
```

The package provides:

- `CanonicalActionParser`: strict canonical JSON or literal-only AST parsing.
- `QwenXmlActionParser`: Qwen `<tool_call>` parsing for GUI, shell, registered
  tools, and termination while retaining canonical actions internally.
- `CanonicalActionExecutor`: coordinate conversion and action routing.
- `ToolRegistry`: explicit tool definitions, JSON-schema validation, and handlers.
- `CanonicalEpisodeRunner`: screenshot/action/observation model loop and logging.
- `OpenAICompatibleModel`: optional multimodal chat-completions adapter.
- `build_tool_registry`: the practical `cua-core-v1` structured-tool profile.

## Practical tool profile

`build_tool_registry()` enables six tools by default:

- `ui.inspect`: query normalized accessibility-tree nodes.
- `filesystem.list`: list or find files under task-approved guest roots.
- `filesystem.read_text`: read bounded text slices.
- `filesystem.write_text`: controlled create, overwrite, or append operations.
- `document.extract_text`: extract PDF, Office, CSV, HTML, and text content.
- `archive.extract`: safely extract ZIP and TAR archives.

File tools require each benchmark adapter to return `ToolPathRoot` entries.
Paths are resolved inside the guest, and traversal or symlink escapes are
rejected. File mutations use a fixed guest helper rather than model-provided
host commands. The default `cua-core-v1` profile and its limits are built into
the package; construct it programmatically:

```python
from cuamaster.runtime import build_tool_registry

registry = build_tool_registry("cua-core-v1")
```

A custom tool catalog can be loaded from a JSON file with
`load_tool_registry(path)`.

## Benchmark integration

A benchmark only needs to implement `RuntimeEnvironment`. The shared runtime
never imports the benchmark or its VM/controller classes.

```python
from cuamaster.runtime import CallbackEnvironment, CanonicalEpisodeRunner

environment = CallbackEnvironment(
    screenshot=lambda: benchmark_env.get_screenshot(),
    screen_size=lambda: benchmark_env.get_screen_size(),
    gui=lambda command, pause: benchmark_env.execute_gui(command, pause=pause),
    shell=lambda command, cwd, timeout: benchmark_env.execute_guest_shell(
        command, cwd=cwd, timeout=timeout
    ),
    terminate=lambda status: benchmark_env.finish(status),
    platform=lambda: benchmark_env.platform,
    accessibility_tree=lambda: benchmark_env.get_accessibility_tree(),
    tool_roots=lambda: benchmark_env.get_tool_roots(),
    default_shell_cwd=lambda: benchmark_env.default_workspace,
)

runner = CanonicalEpisodeRunner(model=model_client, tool_registry=registry)
runner.run(task_instruction, environment, output_dir)
```

The runtime defaults to the Qwen XML tool-call protocol while retaining
canonical actions internally. The equivalent explicit configuration is:

```python
runner = CanonicalEpisodeRunner(
    model=model_client,
    tool_registry=registry,
    action_protocol="qwen_xml",
)
```

Collection and conversion pipelines pass `action_protocol=qwen_xml` explicitly,
and external integrations can pin the same setting with
`CUA_ACTION_PROTOCOL=qwen_xml`. Legacy canonical model output remains available
only through the explicit `action_protocol="canonical"` compatibility setting.

`action_protocol="auto"` accepts either Qwen XML or canonical JSON/Code during
migration. Qwen XML supports `computer_use`, `shell_exec`, generic `tool_call`,
and direct registered-tool aliases such as `filesystem_read_text`. Multiple XML
calls are executed sequentially and logged as individual canonical actions;
the common `mouse_move` + `left_click_drag` pair is normalized to one drag.

The adapter boundary is deliberate:

- GUI execution remains inside the benchmark-controlled desktop environment.
- `shell.exec` must use a guest/VM controller, never the benchmark host shell.
- Tool handlers must be registered explicitly; dynamic imports are unsupported.
- Canonical coordinates stay in `0-1000` space until executor dispatch.
- Tool roots are per-task capabilities, not benchmark-host filesystem paths.

WeaveBench and OSWorld implement this interface with thin adapters. WeaveBench
exposes `/tmp_workspace`; OSWorld exposes `/home/user/Desktop` and
`/home/user/Downloads`. Other benchmarks can reuse the parser, executor, tools,
model loop, and logging without copying them.
