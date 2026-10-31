"""Deterministic variation directives that differentiate compose variants."""
from __future__ import annotations

from typing import Any

from cuamaster.contracts import content_digest


# Axes that change an executable, evaluator-observable dimension of the task.
EXECUTABLE_AXES: tuple[dict[str, str], ...] = (
    {
        "axis": "parameterization",
        "directive": (
            "Change concrete parameters of the goal (file names, target values, "
            "formats, cell ranges, or labels) to different values the supplied "
            "capabilities and evaluator can still verify."
        ),
    },
    {
        "axis": "initial_state",
        "directive": (
            "Assume a different but reachable starting state (application already "
            "open or closed, a pre-existing distractor file or window) and state "
            "it in the instruction when the agent must handle it."
        ),
    },
    {
        "axis": "constraint",
        "directive": (
            "Add one extra requirement on the final state that the supplied "
            "evaluator fragments already observe, making the goal stricter."
        ),
    },
    {
        "axis": "goal_ordering",
        "directive": (
            "Reorder independent sub-goals into a different valid dependency "
            "order and phrase the instruction around that order."
        ),
    },
    {
        "axis": "scope_reduction",
        "directive": (
            "Narrow the goal to a stricter subset (a specific sheet, folder, "
            "message, or object) that the evaluator can still verify."
        ),
    },
)

# Axes that keep the scored outcome identical; safe for source_complete
# task-class recipes whose trusted evaluator fixes the outcome.
OUTCOME_PRESERVING_AXES: tuple[dict[str, str], ...] = (
    {
        "axis": "scenario_framing",
        "directive": (
            "Frame the same goal inside a realistic user scenario (preparing a "
            "report, cleaning up a shared folder, organizing media, replying to "
            "a colleague) without adding or removing scored requirements."
        ),
    },
    {
        "axis": "outcome_first",
        "directive": (
            "State the required end state up front and leave the procedure "
            "entirely to the agent; do not enumerate steps."
        ),
    },
    {
        "axis": "terse_expert",
        "directive": (
            "Write the instruction as a terse request from an expert user who "
            "names exact targets but no procedure."
        ),
    },
    {
        "axis": "verbose_novice",
        "directive": (
            "Write the instruction as a novice user describing what they want in "
            "plain language, mentioning visible UI landmarks but no shortcuts."
        ),
    },
    {
        "axis": "context_motivation",
        "directive": (
            "Give a one-sentence motivation for why the outcome is needed, then "
            "state the exact requirements."
        ),
    },
)

_HARDER_AXES = ("constraint", "initial_state", "goal_ordering")
_EASIER_AXES = ("parameterization", "scope_reduction")

# Scaffolding tiers steer difficulty for outcome-preserving variants without
# touching the scored outcome: the trusted evaluator stays verbatim and only
# the guidance density of the instruction changes.
_LOW_SCAFFOLD_AXES = ("terse_expert", "outcome_first")
_HIGH_SCAFFOLD_AXES = ("verbose_novice", "context_motivation")

_HIGH_SCAFFOLD_DIRECTIVE = (
    " Include brief orienting guidance: name the applications involved and the "
    "order of the major sub-goals, without prescribing exact clicks, menus, or "
    "keyboard shortcuts."
)
_LOW_SCAFFOLD_DIRECTIVE = (
    " State only the required end state; give no procedural hints, application "
    "walkthroughs, or sub-goal ordering."
)


def variation_directive(
    recipe_key: str,
    variant_index: int,
    *,
    outcome_preserving_only: bool = False,
    difficulty_target: str = "",
) -> dict[str, Any]:
    """Pick a deterministic variation directive for one compose variant.

    Axes rotate per (recipe_key, variant_index): a recipe-keyed offset picks
    the starting axis and consecutive variant indices walk the pool round-robin
    so every anchor covers all axes before repeating. Different
    (recipe_key, variant_index) pairs still map to distinct payloads, so API
    cache keys never collide.
    """
    scaffolding = "neutral"
    extra_directive = ""
    if outcome_preserving_only:
        pool = list(OUTCOME_PRESERVING_AXES)
        if difficulty_target == "harder":
            filtered = [item for item in pool if item["axis"] in _LOW_SCAFFOLD_AXES]
            pool = filtered or pool
            scaffolding = "low"
            extra_directive = _LOW_SCAFFOLD_DIRECTIVE
        elif difficulty_target == "easier":
            filtered = [item for item in pool if item["axis"] in _HIGH_SCAFFOLD_AXES]
            pool = filtered or pool
            scaffolding = "high"
            extra_directive = _HIGH_SCAFFOLD_DIRECTIVE
    else:
        pool = list(EXECUTABLE_AXES) + list(OUTCOME_PRESERVING_AXES)
        if difficulty_target == "harder":
            pool = [item for item in pool if item["axis"] in _HARDER_AXES] or pool
        elif difficulty_target == "easier":
            pool = [item for item in pool if item["axis"] in _EASIER_AXES] or pool
    offset = int(content_digest({"recipe": recipe_key})[:8], 16) % len(pool)
    chosen = pool[(offset + variant_index) % len(pool)]
    return {
        "axis": chosen["axis"],
        "directive": chosen["directive"] + extra_directive,
        "variant_index": variant_index,
        "difficulty_target": difficulty_target or "neutral",
        "scaffolding": scaffolding,
        "outcome_preserving": outcome_preserving_only
        or chosen in OUTCOME_PRESERVING_AXES,
    }
