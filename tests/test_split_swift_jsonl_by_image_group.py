from __future__ import annotations

from scripts.split_swift_jsonl_by_image_group import group_from_row, split_rows


def row(app: str, task: str, step: int) -> dict:
    return {
        "messages": [{"role": "user", "content": "<image>"}],
        "images": [f"/data/{app}/{task}/step_{step}.png"],
    }


def test_split_rows_keeps_tasks_disjoint_and_apps_represented() -> None:
    rows = []
    for app in ("chrome", "writer"):
        for task_index in range(4):
            rows.extend(row(app, f"task-{task_index}", step) for step in range(task_index + 1))

    train, val, stats = split_rows(rows, val_ratio=0.2, seed=7)

    train_groups = {group_from_row(item) for item in train}
    val_groups = {group_from_row(item) for item in val}
    assert train_groups.isdisjoint(val_groups)
    assert {app for app, _ in val_groups} == {"chrome", "writer"}
    assert stats["groups"] == 8
    assert stats["train_rows"] + stats["val_rows"] == len(rows)
