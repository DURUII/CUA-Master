"""Inspect and statically verify ScaleCUA-style task specifications."""
from __future__ import annotations

import argparse
import json

from .task_specs import discover_task_specs, verify_task_spec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect versioned GUI task specifications.")
    parser.add_argument("root")
    parser.add_argument("--capability", action="append", default=[])
    parser.add_argument("--require-evaluator", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = []
    failed = False
    for task in discover_task_specs(args.root):
        result = verify_task_spec(task, available_capabilities=args.capability)
        if not args.require_evaluator and result["errors"] == ["missing executable evaluator"]:
            result["errors"] = []
            result["valid"] = True
        failed = failed or not result["valid"]
        results.append(result)
    print(json.dumps({"tasks": results, "count": len(results)}, indent=2, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
