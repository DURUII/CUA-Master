"""统计 JSONL 数据集中每个 application / category / subcategory 的视频数量并打印。

支持传入多个文件，结果会合并后统一统计。

    PYTHONPATH=src python -m video2traj.count data/after/video_data1.jsonl
"""

import argparse
import json
from collections import Counter


def count(paths: list[str]) -> tuple[Counter, Counter, Counter]:
    app_counter: Counter = Counter()
    cat_counter: Counter = Counter()
    sub_counter: Counter = Counter()
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                app_counter[obj.get("application", "<none>")] += 1
                cat = obj.get("category", "<none>")
                sub = obj.get("subcategory", "<none>")
                cat_counter[cat] += 1
                sub_counter[(cat, sub)] += 1
    return app_counter, cat_counter, sub_counter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="输入的 JSONL 文件路径（可多个，合并统计）")
    args = parser.parse_args()

    app_counter, cat_counter, sub_counter = count(args.paths)
    total = sum(app_counter.values())

    print(f"total_videos={total}  apps={len(app_counter)}  "
          f"categories={len(cat_counter)}  subcategories={len(sub_counter)}")

    print("\n== category ==")
    print("rank\tcount\tcategory")
    for rank, (cat, n) in enumerate(cat_counter.most_common(), 1):
        print(f"{rank}\t{n}\t{cat}")

    print("\n== subcategory ==")
    print("rank\tcount\tcategory / subcategory")
    for rank, ((cat, sub), n) in enumerate(sub_counter.most_common(), 1):
        print(f"{rank}\t{n}\t{cat} / {sub}")

    print("\n== application ==")
    print("rank\tcount\tapplication")
    for rank, (app, n) in enumerate(app_counter.most_common(), 1):
        print(f"{rank}\t{n}\t{app}")


if __name__ == "__main__":
    main()
