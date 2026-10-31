# Atomic CUA Action Space

This document defines the canonical hybrid action space (paper Table 7) used by
the CUAMaster pipeline and CUA execution/evaluation code. The implementation
lives in `cuamaster/action_space.py`.

The design is intentionally small and atomic. GUI actions model primitive human
input, `shell.exec` models one sandboxed command execution, and `tool.call`
models one registered structured tool call. Higher-level application behavior
should be represented as a sequence of these actions, not as application-specific
semantic actions.

The GUI primitives also include triple-click, key-down/key-up, and
mouse-down/mouse-up so Qwen-style computer-use calls can be represented without
loss before execution. Screenshot requests remain observations rather than a
canonical action; the Qwen XML compatibility parser converts them to a short
wait because every runtime step captures a fresh screenshot automatically.

## Model Output

Each model step emits one action:

```json
{
  "thought": "optional for training",
  "action": {
    "type": "computer.click",
    "args": {"x": 512, "y": 384, "button": "left"}
  }
}
```

Coordinates use normalized `0-1000` screen space. The executor maps normalized
coordinates to the current screen resolution. Raw pixel coordinates may be saved
as metadata, but the model-facing action should use normalized coordinates.

Screenshots are observations returned by the environment after each step.
`computer.screenshot` is not an action.

## GUI Actions

| Action | Args | Purpose |
| --- | --- | --- |
| `computer.click` | `x,y,button=left/right/middle` | Single click, right click, middle click |
| `computer.double_click` | `x,y,button=left` | Double-click activation or selection |
| `computer.triple_click` | `x,y,button=left` | Triple-click text or row selection |
| `computer.move` | `x,y` | Hover when it changes UI state |
| `computer.drag` | `x1,y1,x2,y2,button=left,duration_ms?` | Drag, slider, selection, mobile swipe |
| `computer.scroll` | `dx,dy,x?,y?` | Vertical or horizontal scrolling |
| `computer.type` | `text` | Text input |
| `computer.press` | `keys[]` | Single key or hotkey |
| `computer.key_down` | `key` | Hold one keyboard key |
| `computer.key_up` | `key` | Release one keyboard key |
| `computer.mouse_down` | `x?,y?,button=left` | Hold one mouse button |
| `computer.mouse_up` | `x?,y?,button=left` | Release one mouse button |
| `computer.wait` | `seconds` | Wait for loading, animation, or async results |

`computer.press` represents both single-key and hotkey input:

```json
{"type": "computer.press", "args": {"keys": ["enter"]}}
{"type": "computer.press", "args": {"keys": ["ctrl", "shift", "t"]}}
```

Keeping one keyboard action avoids splitting equivalent keyboard execution into
separate `key` and `hotkey` labels. The executor should press keys in order and
release them in reverse order.

## Non-GUI Actions

| Action | Args | Purpose |
| --- | --- | --- |
| `shell.exec` | `command,cwd?,timeout_sec?` | Run one controlled shell command |
| `tool.call` | `name,args` | Call one registered external tool |
| `finish` | `status=success/failure,answer?` | End the task |

`shell.exec` should run inside the active sandbox/approval policy. Network
access, writes outside the workspace, destructive commands, privilege escalation,
and sensitive reads should require approval or be rejected.

`tool.call` is a generic structured-tool entry point. Tool definitions should be
stable and auditable:

```json
{
  "name": "stable_action_name",
  "description": "when to use this tool",
  "input_schema": {},
  "output_schema": {},
  "read_only": true,
  "destructive": false,
  "open_world": false
}
```

Use tools for stable environment capabilities such as search, URL fetches,
database queries, clipboard access, or asking the user. Avoid application-level
semantic tools such as `libreoffice_set_column_values` or `complete_checkout`;
those hide planning inside the action space and make video labels harder to
align across data sources.

## Compatibility Mapping

`cuamaster/action_space.py` provides `from_legacy_action(...)` for importing
common OpenCUA/Qwen-style logs:

| Legacy action | Canonical action |
| --- | --- |
| `click`, `left_click` | `computer.click` with `button="left"` |
| `rightClick`, `right_click` | `computer.click` with `button="right"` |
| `middleClick`, `middle_click` | `computer.click` with `button="middle"` |
| `doubleClick`, `double_click` | `computer.double_click` |
| `tripleClick`, `triple_click` | `computer.triple_click` |
| `moveTo`, `move`, `mouse_move` | `computer.move` |
| `dragTo`, `drag`, `left_click_drag` | `computer.drag` |
| `scroll`, `hscroll` | `computer.scroll` |
| `write`, `type` | `computer.type` |
| `press`, `key`, `hotkey` | `computer.press` |
| `keyDown`, `key_down` | `computer.key_down` |
| `keyUp`, `key_up` | `computer.key_up` |
| `mouseDown`, `mouse_down` | `computer.mouse_down` |
| `mouseUp`, `mouse_up` | `computer.mouse_up` |
| `wait` | `computer.wait` |
| `cli_command` | `shell.exec` |
| `api_call` | `tool.call` |
| `terminate` | `finish` |

New annotations should be emitted directly in the canonical schema. The legacy
mapper is for import and migration only.

## Validation Rules

The validator is strict by design:

- Unknown action types are rejected.
- Unknown args are rejected.
- Coordinates must be numbers in `[0, 1000]`.
- Mouse buttons must be `left`, `right`, or `middle`.
- `computer.press.keys` must be a non-empty list of non-empty strings.
- `computer.key_down.key` and `computer.key_up.key` must be non-empty strings.
- Mouse-down/up coordinates are optional, but `x` and `y` must appear together.
- `shell.exec.command` must be a non-empty string.
- `tool.call.args` must be an object.
- `finish.status` must be `success` or `failure`.

These checks catch noisy generated labels early and make downstream adapters
easier to maintain.

## Maintenance Guidelines

- Add a new action type only when it cannot be represented cleanly by existing
  primitives and is needed across multiple environments.
- Prefer executor/adaptor improvements over action-space expansion.
- Do not add screenshot/observe actions unless the environment model changes;
  observations are returned automatically after each step.
- Keep app-specific workflow shortcuts out of the core action space.
- Keep `ACTION_SPECS` as the source of truth for action names, categories,
  required args, and optional args.
- Update `tests/test_action_space.py` whenever action args, mappings, or
  validation rules change.
