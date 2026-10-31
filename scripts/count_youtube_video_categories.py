#!/usr/bin/env python3
"""统计 youtube_video 目录中每个一级类和二级子类的视频数量。"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("data/videos")
DEFAULT_OUTPUT = Path("results/youtube_video_category_counts.csv")
DEFAULT_MARKDOWN_OUTPUT = Path("results/youtube_video_category_counts.md")
DEFAULT_VIDEO_EXTENSIONS = {
    ".avi",
    ".flv",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}


@dataclass(frozen=True)
class CountRow:
    category: str
    subcategory: str
    count: int
    row_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="统计两层目录结构中每个类/子类的视频数量。"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"待统计目录，默认：{DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"CSV 输出路径，默认：{DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=DEFAULT_MARKDOWN_OUTPUT,
        help=f"Markdown 输出路径，默认：{DEFAULT_MARKDOWN_OUTPUT}",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="统计所有文件，而不只统计常见视频格式。",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="包含名称以点号开头的隐藏类和子类目录。",
    )
    return parser.parse_args()


def visible_directories(path: Path, include_hidden: bool) -> list[Path]:
    directories = [
        child
        for child in path.iterdir()
        if child.is_dir() and (include_hidden or not child.name.startswith("."))
    ]
    return sorted(directories, key=lambda item: item.name.casefold())


def count_files(path: Path, extensions: set[str] | None) -> int:
    count = 0
    pending = [path]

    while pending:
        current = pending.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    if extensions is None or Path(entry.name).suffix.lower() in extensions:
                        count += 1

    return count


def collect_counts(
    root: Path,
    extensions: set[str] | None,
    include_hidden: bool,
) -> list[CountRow]:
    rows: list[CountRow] = []
    grand_total = 0

    for category_dir in visible_directories(root, include_hidden):
        category_total = 0
        subcategory_dirs = visible_directories(category_dir, include_hidden)

        for subcategory_dir in subcategory_dirs:
            file_count = count_files(subcategory_dir, extensions)
            category_total += file_count
            rows.append(
                CountRow(
                    category=category_dir.name,
                    subcategory=subcategory_dir.name,
                    count=file_count,
                    row_type="子类",
                )
            )

        direct_file_count = sum(
            1
            for child in category_dir.iterdir()
            if child.is_file()
            and (extensions is None or child.suffix.lower() in extensions)
        )
        if direct_file_count:
            category_total += direct_file_count
            rows.append(
                CountRow(
                    category=category_dir.name,
                    subcategory="（一级类目录直接文件）",
                    count=direct_file_count,
                    row_type="子类",
                )
            )

        rows.append(
            CountRow(
                category=category_dir.name,
                subcategory="合计",
                count=category_total,
                row_type="一级类合计",
            )
        )
        grand_total += category_total

    rows.append(CountRow(category="全部", subcategory="合计", count=grand_total, row_type="总计"))
    return rows


def write_csv(rows: Iterable[CountRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["一级类", "二级子类", "数量", "行类型"])
        for row in rows:
            writer.writerow([row.category, row.subcategory, row.count, row.row_type])


def escape_markdown(value: str) -> str:
    return value.replace("|", "\\|")


def write_markdown(rows: Iterable[CountRow], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("| 一级类 | 二级子类 | 数量 | 行类型 |\n")
        file.write("|---|---|---:|---|\n")
        for row in rows:
            file.write(
                f"| {escape_markdown(row.category)} | "
                f"{escape_markdown(row.subcategory)} | {row.count} | {row.row_type} |\n"
            )

        category_sub_values: dict[str, list[int]] = {}
        for row in rows:
            if row.row_type == "子类":
                category_sub_values.setdefault(row.category, []).append(row.count)

        file.write("\n## 子类数量数组\n\n")
        file.write("每一行对应一个一级类，每个数值对应表格中同一顺序的二级子类数量。\n\n")
        file.write("```python\n")
        file.write("sub_values = [\n")
        for category, values in category_sub_values.items():
            file.write(f"    {values},  # {category}\n")
        file.write("]\n")
        file.write("```\n")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"目录不存在或不是目录：{root}")

    extensions = None if args.all_files else DEFAULT_VIDEO_EXTENSIONS
    rows = collect_counts(root, extensions, args.include_hidden)
    write_csv(rows, args.output)
    write_markdown(rows, args.markdown_output)

    category_count = sum(row.row_type == "一级类合计" for row in rows)
    subcategory_count = sum(row.row_type == "子类" for row in rows)
    grand_total = rows[-1].count
    print(f"一级类数量：{category_count}")
    print(f"二级子类数量：{subcategory_count}")
    print(f"视频总数：{grand_total}")
    print(f"CSV 结果：{args.output.resolve()}")
    print(f"Markdown 结果：{args.markdown_output.resolve()}")


if __name__ == "__main__":
    main()
