"""CLI for listing, verifying, and registering VM images."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .image_store import ImageStore

DEFAULT_STORE = Path(__file__).resolve().parents[3] / "vm_images"


def _software(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"software must be NAME=VERSION: {value}")
        name, version = value.split("=", 1)
        result[name.strip()] = version.strip()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="cuamaster-vm-images")
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("verify")
    register = commands.add_parser("register")
    register.add_argument("--id", required=True)
    register.add_argument("--source", required=True)
    register.add_argument("--os-type", required=True)
    register.add_argument("--os-version", required=True)
    register.add_argument("--architecture", default="x86_64")
    register.add_argument("--snapshot", default="clean")
    register.add_argument("--derived-from", default=None)
    register.add_argument("--software", action="append", default=[])
    register.add_argument(
        "--mode",
        choices=["link", "hardlink", "copy", "reference", "adopt"],
        default="link",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    store = ImageStore(args.store)
    if args.command == "list":
        for record in store.list():
            print(json.dumps(record.as_metadata(), ensure_ascii=False))
        return 0
    if args.command == "verify":
        failed = 0
        for record in store.list():
            try:
                store.verify_record(record)
                descriptor = json.loads(
                    record.descriptor_path.read_text(encoding="utf-8")
                )
                inventory_status = descriptor.get("inventory_status", "complete")
                print(
                    f"OK  {record.id}  {record.path}  "
                    f"inventory={inventory_status}"
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"ERR {record.id}  {exc}")
        return 1 if failed else 0
    record = store.register(
        image_id=args.id,
        source=args.source,
        os_type=args.os_type,
        os_version=args.os_version,
        architecture=args.architecture,
        snapshot=args.snapshot,
        software=_software(args.software),
        mode=args.mode,
        derived_from=args.derived_from,
    )
    print(json.dumps(record.as_metadata(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
