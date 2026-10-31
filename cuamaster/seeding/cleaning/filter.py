"""Flag or drop records whose title/description don't mention their application.

The source search matches by keyword, so results can drift off-topic (a generic
word like "Wave" pulling in unrelated videos) or match nothing real (a made-up
app name filling up with noise). This scores each record by whether its target
application name actually appears in the title or description, then either
labels every record with that flag or drops the ones that miss.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("video2traj")


def _core_and_tokens(application: str) -> tuple[str, list[str]]:
    """Lowercased app name with parenthetical stripped, plus its meaningful tokens.

    e.g. "GLM (General Lake Model)" -> ("glm", ["glm"]);
         "Adobe After Effects" -> ("adobe after effects", ["adobe", "after", "effects"]).
    """
    core = re.sub(r"\(.*?\)", "", application).strip().lower()
    tokens = [t for t in re.split(r"[\s/&.+-]+", core) if len(t) > 2]
    return core, tokens


def title_matches(
    application: str,
    title: str | None,
    description: str | None,
    loose: bool = False,
) -> bool:
    """True if the app name (as a phrase, or all its tokens) appears in title/desc.

    With loose=True, a match also succeeds when only the first (core) token
    appears, so multi-word app names like "ADP Workforce Now" still match
    titles that mention just "ADP".
    """
    hay = " ".join([title or "", description or ""]).lower()
    core, tokens = _core_and_tokens(application)
    if core and core in hay:
        return True
    if tokens and all(t in hay for t in tokens):
        return True
    if loose and tokens and tokens[0] in hay:
        return True
    return False


def run(input_path: str, output_path: str, drop_unmatched: bool, loose: bool = False) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    matched = 0
    with open(output, "w", encoding="utf-8") as out:
        for line in open(input_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            total += 1
            record = json.loads(line)
            ok = title_matches(
                record.get("application", ""),
                record.get("title"),
                record.get("description"),
                loose=loose,
            )
            if ok:
                matched += 1
            if drop_unmatched:
                if ok:
                    out.write(line + "\n")
            else:
                record["title_match"] = ok
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    dropped = total - matched
    if drop_unmatched:
        logger.info(
            "done: %d read, %d kept, %d dropped (%.1f%%) -> %s",
            total, matched, dropped, dropped / total * 100 if total else 0, output,
        )
    else:
        logger.info(
            "done: %d read, %d labeled title_match=true, %d false (%.1f%%) -> %s",
            total, matched, dropped, dropped / total * 100 if total else 0, output,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flag or drop records whose title/description omit the application name."
    )
    parser.add_argument("--input", default="data/merged.jsonl", help="input JSONL path")
    parser.add_argument(
        "--output",
        default=None,
        help="output JSONL path; default: <input>.labeled.jsonl, or <input>.filtered.jsonl with --drop-unmatched",
    )
    parser.add_argument(
        "--drop-unmatched",
        action="store_true",
        help="drop records that miss instead of labeling every record with title_match",
    )
    parser.add_argument(
        "--loose",
        action="store_true",
        help="also match when only the app's first (core) token appears, so "
             "multi-word names like 'ADP Workforce Now' match titles saying just 'ADP'",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_path = args.output
    if output_path is None:
        p = Path(args.input)
        suffix = "filtered" if args.drop_unmatched else "labeled"
        output_path = str(p.with_name(f"{p.stem}.{suffix}{p.suffix}"))

    run(args.input, output_path, args.drop_unmatched, loose=args.loose)


if __name__ == "__main__":
    main()
