"""Clean the video_data*.jsonl QC outputs: cross-file dedup + noise filter.

Two operations, in order:
1. Cross-file dedup by video_id. First occurrence wins (inputs processed in the
   order given), later duplicates are dropped.
2. Extended GUI-relevance noise filter. Builds on the shared noise regexes and
   additionally drops commercials/ads, music videos (M/V), official audio,
   nursery rhymes / kids songs, karaoke / sing-along clips, channel/search
   results that are not videos, pronunciation/definition clips, and a small set
   of high-risk application-name collisions where the title is clearly about a
   physical object, game, song, or general concept rather than software.
   Titles that are real software tutorials are exempted and kept.

    PYTHONPATH=src python -m video2traj.clean
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

from ..common.noise import EXEMPT, GAMEPLAY, MOVIETV, PRONOUNCE

# Extended superset of noise.MUSIC: also catches official audio, M/V, kids
# songs, karaoke, and sing-along clips.
MUSIC = re.compile(
    r"(?i)(cover song|piano cover|guitar cover|drum cover|\blive (?:at|in|from)\b"
    r"|in concert|full album|official (?:music )?video|official lyric|\blyrics?\b"
    r"|official audio|\bm/v\b|\bmusic video\b|\bremix\b"
    r"|nursery rhyme|kids? song|songs? for kids|children'?s song"
    r"|karaoke|sing[- ]?along"
    r"|grade \d+ (?:piano|keyboard|guitar|drum|violin|rock|pop))"
)
COMMERCIAL = re.compile(
    r"(?i)(\bcommercial\b|\badvertisement\b|\btv (?:ad|spot)\b|official tv ad)"
)
EXTRA_NOISE = re.compile(
    r"(?i)(\basmr\b"
    r"|\bvlog\b|day in (?:my|the) life"
    r"|reacts? to|reacting to|\breaction\b"
    r"|breaking news|\bnews\b"
    r"|compilation|\btop \d+\b|best of"
    r"|\bprank\b)"
)
VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
YOUTUBE_WATCH = re.compile(r"^https://www\.youtube\.com/watch\?v=[A-Za-z0-9_-]{11}(?:&|$)")
PURE_PRONUNCIATION = re.compile(
    r"(?i)(^\s*how to say\b|\bhow to say [\"']|high quality voices"
    r"|\bmeaning, definition & pronunciation\b|\bpronunciation guide\b"
    r"|^\s*what does .{1,40} mean\??\s*$|\b- how to say\b|\bhow to say .{1,40}!?.*#)"
)

SOFTWARE_HINT = re.compile(
    r"(?i)(\bapp\b|\bapps\b|software|website|web site|online|login|sign in|account"
    r"|install|setup|configure|dashboard|platform|portal|plugin|extension|api|code"
    r"|python|javascript|docker|server|crm|chart|diagram|database|gui|windows|linux|mac)"
)

APP_COLLISION_RULES = {
    "amber": re.compile(
        r"(?i)(quadrobics|brawl stars|genshin|guitar|piano|procreate|mold review|faux amber"
        r"|how to draw|charged shot|\b311\b)"
    ),
    "arise": re.compile(
        r"(?i)(arise crossover|solo leveling|tales of arise|ragnarok|noob to god|noob to pro"
        r"|new players|guitar|cover by|chords|gospel chords|roblox|skip the entire game)"
    ),
    "bark": re.compile(
        r"(?i)(bark like a dog|tree bark|birch tree|watercolor|paint|painting)"
    ),
    "booklore": re.compile(
        r"(?i)(how to say|podcast|book summary|letter r|alphabetbook|congratulates|book opinions)"
    ),
    "dunst": re.compile(r"(?i)(kirsten dunst|oil painting)"),
    "g2": re.compile(
        r"(?i)(e-scooter|kugoo|kukirin|driving test|guitar|pedal|pool apnea|csgo|unboxing)"
    ),
    "idealist": re.compile(
        r"(?i)(metaphysics|philosopher|historicist|realism|anti-realism|eyeshadow|cosplay"
        r"|what does \"?idealist\"? really mean|cultural theory)"
    ),
    "ikea": re.compile(
        r"(?i)(assembly|drawer|dresser|cabinet|bookcase|wardrobe|desk|shelf|kallax|malm"
        r"|brimnes|storklinta|hauga|kleppstad|furniture|follow ikea)"
    ),
    "kayak": re.compile(
        r"(?i)(how to kayak|kayaking|paddle|sea kayak|kayak fishing|kayaker|self-rescue|sweep kayak roll)"
    ),
    "mermaid": re.compile(
        r"(?i)(makeup|how to draw|ariel|little mermaid|mermaid doll|swim like a mermaid"
        r"|mermaids|berry avenue|halloween look)"
    ),
    "nuke": re.compile(r"(?i)(cs2|roblox|build a boat for treasure|play nuke at a high level)"),
    "starfish": re.compile(
        r"(?i)(origami|crochet|how to draw|craft|cardboard|clay|balloon animals|sea animals"
        r"|starfish for kids)"
    ),
    "target": re.compile(
        r"(?i)(sporting clays|shoot|bird target|target audience|vuforia|minecraft|slingshot"
        r"|effect house|camera target|after effect)"
    ),
    "trinity": re.compile(
        r"(?i)(piano|guitar|drum|crochet|necktie|nuface|xbox|muay thai|rock & pop|synthesia)"
    ),
    "tropy": re.compile(r"(?i)(crash bandicoot|vexo iq)"),
    "twenty": re.compile(r"(?i)(twenty one pilots|dance tutorial|wordpress|twenty twenty|guitar|piano|bass|drum)"),
    "volatility": re.compile(
        r"(?i)(market volatility|options?|vix|volatility smile|volatility skew|garch|trading"
        r"|stocks|implied volatility|calculate volatility in excel)"
    ),
    "walmart": re.compile(
        r"(?i)(bass fishing|buying guide|minecraft|crochet kits|unboxing|onn tablet|tv box)"
    ),
}


def reason(record):
    title = record.get("title") or ""
    application = (record.get("application") or "").strip().lower()
    url = record.get("url") or ""
    video_id = record.get("video_id") or ""

    if not VIDEO_ID.match(video_id) or not YOUTUBE_WATCH.match(url):
        return "invalid_video_url"
    if PURE_PRONUNCIATION.search(title):
        return "pronounce"

    if EXEMPT.search(title):
        return None
    app_rule = APP_COLLISION_RULES.get(application)
    if app_rule and app_rule.search(title):
        return "app_name_collision"
    if GAMEPLAY.search(title):
        return "gameplay"
    if MUSIC.search(title):
        return "music"
    if MOVIETV.search(title):
        return "movietv"
    if PRONOUNCE.search(title):
        return "pronounce"
    if COMMERCIAL.search(title):
        return "commercial"
    if EXTRA_NOISE.search(title):
        return "extra_noise"
    return None


def parse_views(record):
    try:
        return int(record.get("view_count"))
    except (TypeError, ValueError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+",
                    default=["data/after/video_data.jsonl",
                             "data/after/video_data2.jsonl"])
    ap.add_argument("--outputs", nargs="+",
                    default=["data/after/video_data.clean.jsonl",
                             "data/after/video_data2.clean.jsonl"])
    args = ap.parse_args()
    assert len(args.inputs) == len(args.outputs)

    # Stage 1: video_id dedup + noise filter; collect survivors in memory.
    seen = set()
    stats = Counter()
    dropped_dup = [0] * len(args.inputs)
    dropped_noise = [0] * len(args.inputs)
    survivors = []  # (file_idx, line_idx, line, norm_title, views)
    for fi, inp in enumerate(args.inputs):
        with open(inp) as fin:
            for li, line in enumerate(fin):
                line = line.rstrip("\n")
                if not line:
                    continue
                r = json.loads(line)
                vid = r.get("video_id")
                if vid in seen:
                    dropped_dup[fi] += 1
                    continue
                why = reason(r)
                if why:
                    dropped_noise[fi] += 1
                    stats[why] += 1
                    continue
                seen.add(vid)
                norm_title = (r.get("title") or "").strip().lower()
                survivors.append((fi, li, line, norm_title, parse_views(r)))

    # Stage 2: dedup by normalized title, keep the highest-view record.
    # Empty titles are never deduped. Ties keep the earliest occurrence.
    best = {}  # norm_title -> index into survivors
    for idx, (fi, li, line, norm_title, views) in enumerate(survivors):
        if not norm_title:
            continue
        cur = best.get(norm_title)
        if cur is None or views > survivors[cur][4]:
            best[norm_title] = idx
    keep_flags = [True] * len(survivors)
    dropped_dup_title = [0] * len(args.inputs)
    for idx, (fi, li, line, norm_title, views) in enumerate(survivors):
        if not norm_title:
            continue
        if best[norm_title] != idx:
            keep_flags[idx] = False
            dropped_dup_title[fi] += 1

    # Stage 3: write survivors back in original (file, line) order.
    outs = [open(o, "w") for o in args.outputs]
    kept = [0] * len(args.inputs)
    for idx, (fi, li, line, norm_title, views) in enumerate(survivors):
        if not keep_flags[idx]:
            continue
        outs[fi].write(line + "\n")
        kept[fi] += 1
    for o in outs:
        o.close()

    for fi, (inp, outp) in enumerate(zip(args.inputs, args.outputs)):
        print(f"{inp} -> {outp}")
        print(f"  kept={kept[fi]} dropped_dup={dropped_dup[fi]} "
              f"dropped_noise={dropped_noise[fi]} "
              f"dropped_dup_title={dropped_dup_title[fi]}")
    print("noise breakdown:")
    for k, v in stats.most_common():
        print(f"  {v}\t{k}")


if __name__ == "__main__":
    main()
