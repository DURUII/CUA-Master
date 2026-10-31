# Benchmark Interaction Harnesses: Integration and Train/Eval Consistency

This document explains how to reuse this repository's GUI, CLI, and hybrid
interaction constraints in another benchmark's evaluate flow, so that the
tools, action protocol, screenshots, and shell execution location the model
sees at evaluation time stay consistent with trajectory collection and SFT
training data.

## 1. Goals

Training data and inference-time evaluation should share one set of constraints:

- the same interaction capabilities: GUI, CLI, or GUI + CLI;
- the same canonical action types;
- the same Claude Code tool allowlist;
- the same GUI coordinates, screenshot timing, and shell execution location;
- the same action protocol, system prompt, and termination semantics;
- no disabling tools during training while re-enabling them at evaluation;
- no agent bypasses through the benchmark evaluator, hidden application state,
  or collector files.

The reusable code lives in:

```text
cuamaster/benchmarks/
├── base.py          # shared capabilities and canonical-action constraints
├── modes.py         # the three harnesses: GUI / CLI / hybrid
├── claude_code.py   # Claude Code tool names and permission mapping
└── prompts.py       # GUI+CLI system prompt shared by training and evaluation
```

The package does not depend on OSWorld, WeaveBench, or the trajectory
collector, so other benchmark adapters can import it directly.

## 2. Terminology

### 2.1 Agent harness

The *agent harness* is the runtime that actually drives the model, for example:

- `claudecode`;
- `openclaw`;
- `codex`;
- `hermes`.

The current WeaveBench configuration selects it with `backend_options.harness`.

### 2.2 Interaction harness

The *interaction harness* is the interaction surface the model is allowed to
use inside the benchmark environment:

- `gui`;
- `cli`;
- `hybrid`.

The current configuration uses:

```yaml
backend_options:
  harness: claudecode
  interaction_harness: hybrid
```

Do not conflate the two. `claudecode + hybrid` means Claude Code runs the
model, but the model may only use the GUI and the in-environment CLI.

### 2.3 Benchmark adapter

The *benchmark adapter* maps the generic capabilities onto a concrete
benchmark:

- capture screenshots and screen size;
- execute mouse, keyboard, and scroll actions;
- execute shell commands inside the environment under test;
- initialize and reset the environment;
- invoke the benchmark's own evaluator;
- persist actions, execution results, and observations.

The interaction harness is not responsible for the benchmark evaluator, and it
must not give the model access to the evaluator.

## 3. The Three Interaction Harnesses

| Harness | Canonical actions | Claude Code guest | Claude Code host |
| --- | --- | --- | --- |
| `gui` | `computer.*`, `finish` | computer MCP | computer MCP |
| `cli` | `shell.exec`, `finish` | `Bash` | guest-shell MCP |
| `hybrid` | `computer.*`, `shell.exec`, `finish` | computer MCP + `Bash` | computer MCP + guest-shell MCP |

None of the three modes allows generic `tool.call`. The following Claude Code
tools are disallowed by default:

- `Read`, `Write`, `Edit`, `Glob`, `Grep`, `NotebookEdit`;
- `WebFetch`, `WebSearch`;
- internal orchestration tools such as `Agent`, `Task*`, `TodoWrite`, `Skill`,
  `AskUserQuestion`.

This prevents training data from containing structured tools whose
implementations the evaluation runtime does not provide.

## 4. Public API

### 4.1 Getting a harness

```python
from cuamaster.benchmarks import get_interaction_harness

interaction = get_interaction_harness("hybrid")

assert interaction.supports_gui
assert interaction.supports_cli
assert interaction.allows_action_type("computer.click")
assert interaction.allows_action_type("shell.exec")
assert not interaction.allows_action_type("tool.call")
```

The available constants are:

```python
from cuamaster.benchmarks import CLI_HARNESS, GUI_HARNESS, HYBRID_HARNESS
```

Training and evaluation should obtain the hybrid prompt from the same entry
point:

```python
from cuamaster.benchmarks import hybrid_system_prompt

system_prompt = hybrid_system_prompt("qwen_xml")
```

That prompt only declares `computer_use` and `shell_exec`. The outer XML tag is
still called `<tool_call>`, but that is Qwen protocol syntax; it does not mean
generic structured tools are enabled.

### 4.2 Validating canonical actions before execution

Every benchmark adapter should run the harness check before an action actually
executes, not merely filter actions when exporting data:

```python
from cuamaster.benchmarks import get_interaction_harness


class BenchmarkActionGateway:
    def __init__(self, interaction_harness: str):
        self.interaction = get_interaction_harness(interaction_harness)

    def execute(self, environment, action):
        if not self.interaction.allows_action_type(action.type):
            raise ValueError(
                f"{self.interaction.name} harness rejects action {action.type!r}"
            )
        if action.type.startswith("computer."):
            return environment.execute_gui(action)
        if action.type == "shell.exec":
            return environment.execute_shell(**action.args)
        if action.type == "finish":
            return environment.finish(action.args["status"])
        raise AssertionError("unreachable action type")
```

When an action is rejected, terminate or mark the episode as invalid. Do not
silently drop the action and keep evaluating, or the model's behavior and its
score lose their correspondence.

### 4.3 Claude Code tool permissions

```python
from cuamaster.benchmarks import claude_code_tool_policy

policy = claude_code_tool_policy("hybrid", runtime_location="guest")

command.extend([
    "--allowedTools",
    policy.allowed_argument,
    "--disallowedTools",
    policy.disallowed_argument,
])
```

The guest hybrid allowlist is:

```text
mcp__weavebench_computer__computer,Bash
```

The host hybrid allowlist is:

```text
mcp__weavebench_computer__computer,mcp__weavebench_computer__shell
```

In host mode, the controller machine's local `Bash` must be disabled; all CLI
commands must be forwarded into the environment under test by the guest-shell
MCP.

Do not combine this with `--dangerously-skip-permissions`. The current guest
Claude compatibility layer removes that flag and writes explicit
`--allowedTools` and `--disallowedTools` instead. If upstream launcher-script
changes make the permission patch unrecognizable, the implementation fails
closed instead of continuing without permission restrictions.

## 5. Current Trajectory-Collection Call Chain

### 5.1 Task generation is not trajectory collection

```bash
cuamaster-synthesize --config <generation.yaml>
```

This command only runs the task-generation stage:

```text
index → decompose → compose → validate
```

It does not start a VM and does not invoke Claude Code.

### 5.2 End-to-end task compilation, collection, and export

The production entry point is:

```bash
python -m scripts.task_trajectory_pipeline \
  --config configs/collection/vm_matrix.example.json \
  --run-dir <generated-task-run> \
  --output-root <output-root>
```

The invocation order is:

```text
prepare taskset
  ↓
check model API
  ↓
cuamaster-collect
  ↓
WeaveBench backend
  ↓
weavebench_runner compatibility layer
  ↓
Claude Code + selected interaction harness
  ↓
canonical trajectory export
  ↓
SFT / visual-only / mixed dataset export
```

The configuration values:

```yaml
harness: claudecode
interaction_harness: hybrid
runtime_location: guest
```

resolve to:

```text
Claude Code runs inside the guest VM
├── computer MCP → GUI actions → computer.*
└── Bash         → guest CLI   → shell.exec
```

### 5.3 Why the hybrid compatibility shim exists

The current upstream WeaveBench `messages_gui` runner does not natively accept
`hybrid`. The compatibility layer:

1. records the requested `interaction_harness=hybrid`;
2. temporarily rewrites the upstream runner's `--mode hybrid` to `--mode gui`
   to enable the GUI prompt, screenshots, and the computer MCP;
3. keeps guest `Bash` available through the Claude Code allowlist;
4. restores the score record and directory semantics to `mode=hybrid` after
   the run.

So this is compatibility with the upstream runner interface, not a fake hybrid
assembled by merging two separate trajectories after the fact. GUI and CLI
calls still happen in the same Claude Code session, the same VM, and the same
episode.

## 6. Integrating Another Benchmark

### 6.1 Recommended architecture

```text
Model client
  ↓
Shared system prompt / action parser
  ↓
InteractionHarness action gate
  ↓
Benchmark adapter
  ├── screenshot / screen size
  ├── execute GUI
  ├── execute guest shell
  ├── reset / setup
  └── finish
  ↓
Canonical trajectory recorder
  ↓
Benchmark evaluator (not accessible to the model)
```

### 6.2 Minimal integration steps

1. Read `interaction_harness` from the configuration.
2. Call `get_interaction_harness()`; fail immediately on unknown values.
3. Register the actual execution tools based on `supports_gui` and
   `supports_cli`.
4. Apply the corresponding tool allowlist before the model starts, instead of
   filtering the transcript afterwards.
5. Normalize all GUI actions to `computer.*`.
6. Normalize all in-environment CLI calls to `shell.exec`.
7. Record screenshots, execution status, and errors before and after each
   action.
8. Run the evaluator independently after the agent finishes; never expose it
   to the model.
9. Save `interaction.as_dict()` in the episode metadata.
10. Re-validate action types before export, as a defensive gate.

### 6.3 Configuration example

```yaml
benchmark: example_benchmark
agent:
  harness: claudecode
  model: example-model
interaction_harness: hybrid
runtime_location: guest
action_protocol: qwen_xml
```

Recommended fields to persist in evaluation artifacts:

```json
{
  "agent_harness": "claudecode",
  "interaction_harness": {
    "name": "hybrid",
    "capabilities": ["cli", "gui"],
    "allows_structured_tools": false
  },
  "runtime_location": "guest",
  "action_protocol": "qwen_xml"
}
```

## 7. Train/Eval Consistency Requirements

### 7.1 The action space must match

If the training set only contains:

```text
computer.*
shell.exec
finish
```

then evaluation may only expose those actions. Do not additionally enable
`Read`, `Write`, browser search, database access, or benchmark-specific tools
at evaluation time.

Conversely, if evaluation only supports GUI, the training data should use the
visual-only bundle; do not train heavily on `shell.exec` and then remove shell
at evaluation time.

### 7.2 The action protocol must match

The current production configuration uses:

```yaml
action_protocol: qwen_xml
```

Training and inference should share the same:

- system prompt;
- XML function names;
- argument names and types;
- one-action-per-step constraint;
- coordinate normalization;
- `finish(success|failure)` semantics.

Do not train with Qwen XML and then switch evaluation to another, unaligned
JSON/tool-call schema.

### 7.3 Observations must match

At minimum, keep these rules consistent:

- every GUI action has a corresponding pre-screenshot;
- the post-action screenshot serves as the next step's observation;
- consecutive shell actions inherit the previously visible state;
- screenshot size and coordinate conversion rules are identical;
- history window length, action-history formatting, and image order are
  identical;
- do not provide extra accessibility trees, DOM, or hidden state at
  evaluation time unless the training data also used them.

### 7.4 The CLI location must match

`shell.exec` must execute inside the benchmark guest environment:

- guest runtime: may map to guest `Bash`;
- host runtime: must map to the guest-shell MCP;
- never execute shell commands on the evaluator or collector host by mistake.

The CLI's default working directory, timeout, stdout/stderr truncation rules,
and exit-code semantics should also stay consistent.

### 7.5 The environment must match

Where possible, use the same values for training collection and evaluation:

- VM image and snapshot;
- OS, desktop resolution, language, and timezone;
- application versions and pre-installed dependencies;
- network proxy and site mappings;
- setup/reset procedures;
- evaluator version.

Environment differences should be written into episode metadata, not inferred
from task directory names alone.

## 8. Quality and Safety Gates

Even when tool permissions are restricted at the source, the evaluation
adapter should still reject:

- `tool.call`;
- empty `computer.type`;
- scrolls with `dx=0, dy=0`;
- plaintext passwords, tokens, and API keys;
- access to evaluator artifacts, score files, or answer files;
- hidden-state shortcuts such as browser profiles, mail stores, or the
  internal structure of Office archives;
- modification of the collector, screenshot hooks, or the benchmark runner;
- local proxy services, network reconfiguration, or deploying services outside
  the task.

Training collection additionally records in `quality.json`:

- `training_tier`;
- `action_policy_findings`;
- `unchanged_wait_steps`;
- execution/observation alignment issues.

Other benchmarks are encouraged to emit the same field names so audits can be
unified across benchmarks.

## 9. Smoke-Test Checklist

After integrating a new benchmark, first run minimal GUI, CLI, and hybrid
tasks separately:

### GUI

- only computer calls are possible;
- Bash and structured tools are rejected;
- every GUI action has pre/post screenshots;
- coordinates land consistently with the actual resolution.

### CLI

- only the guest shell is callable;
- computer and structured tools are unavailable;
- stdout, stderr, exit code, and timeout are recorded correctly;
- commands verifiably run in the guest, not on the controller host.

### Hybrid

- computer and guest shell can alternate within one session;
- transcript order matches canonical trajectory order;
- Bash exports as `shell.exec`, computer exports as `computer.*`;
- no generic `tool.call` is produced;
- score, episode metadata, and output directories all record `hybrid`.

## 10. Recommended Evaluation Acceptance Criteria

Before using a new benchmark for formal model comparison, satisfy at least:

1. the interaction harness takes effect before the model starts;
2. `--dangerously-skip-permissions` is not used to bypass the allowlist;
3. all canonical actions pass `allows_action_type()`;
4. there is no `tool.call`;
5. shell is confirmed to run in the guest;
6. the GUI screenshot state chain is continuous;
7. the evaluator is invisible to the agent;
8. training and evaluation use the same action protocol;
9. metadata records the agent harness, interaction harness, runtime location,
   and environment versions;
10. at least one set of GUI, CLI, and hybrid smoke tests has completed.

## 11. Code Entry Points

- Generic harness: `cuamaster/benchmarks/base.py`
- Registry of the three modes: `cuamaster/benchmarks/modes.py`
- Claude Code permission mapping: `cuamaster/benchmarks/claude_code.py`
- WeaveBench backend integration: `cuamaster/collection/backends/weavebench.py`
- Claude Code guest compatibility: `cuamaster/collection/weavebench_runner.py`
- Claude Code host adapter: `cuamaster/collection/host_claude_agent.py`
- Canonical action schema: `cuamaster/action_space.py`
- Canonical runtime: `cuamaster/runtime/`
- Trajectory quality gates: `cuamaster/collection/quality.py`

A new benchmark should get its own adapter rather than copying and modifying
the Claude Code allowlist or the GUI/CLI mode logic. Shared logic belongs in
`cuamaster.benchmarks` or `cuamaster.runtime`; the benchmark directory should
only contain environment startup, action execution, and evaluator differences.
