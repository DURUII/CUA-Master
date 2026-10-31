"""Drop records that obviously cannot contain a software GUI operation trajectory.

Conservative filter for the CUA dataset: only removes titles that clearly are
game playthroughs, music performances/MVs, movie/TV trailers or clips, and
pronunciation lessons. Game-titled records that are actually software tutorials
(how to install/use/configure, "tutorial", "step by step", "getting started")
are exempted and kept.

    PYTHONPATH=src python -m video2traj.noise_filter
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from ..common.noise import basic_reason


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/merged.filtered.jsonl")
    ap.add_argument("--output", default="data/merged.filtered.filterd.jsonl")
    args = ap.parse_args()

    kept = dropped = 0
    stats: Counter = Counter()
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            r = json.loads(line)
            why = basic_reason(r.get("title") or "")
            if why:
                dropped += 1
                stats[why] += 1
            else:
                fout.write(line + "\n")
                kept += 1

    print(f"input={kept + dropped} kept={kept} dropped={dropped}")
    for k, v in stats.most_common():
        print(f"  {v}\t{k}")


if __name__ == "__main__":
    main()
