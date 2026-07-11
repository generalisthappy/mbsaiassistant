#!/usr/bin/env python3
"""
Join extracted explanatory notes (mbs_notes.jsonl) to schedule items, mental-health
items first, writing associations into mbs_chunks.jsonl:

    record["notes"] = {
        "primary":   [{note_id, title}, ...],   # item named in the note title
        "mentioned": [{note_id, title}, ...],    # item cross-referenced in the body
    }

Primary links are authoritative (the note is ABOUT the item). Full note bodies stay
in mbs_notes.jsonl and are looked up by note_id at answer time.

Pipeline order: mbs_parser.py (XML -> chunks) -> mbs_notes_parser.py (docx -> notes)
-> attach_notes.py (this). Re-running the XML parser clears the notes field, so run
this again after.

Usage:
    python attach_notes.py --chunks mbs_chunks.jsonl --notes mbs_notes.jsonl \
        --map mbs_profession_map.json
"""

import argparse
import json
import re
from collections import defaultdict

# Note IDs encode the group they govern: <Letter>N.<GroupNumber>.<Seq>
#   AN.36.x -> group A36 (category 1)   MN.6.x -> group M6 (category 8)
# The letter maps to a category; ".0." notes are category-general (not group-
# specific) and reach items via their explicit item references instead.
_NOTE_ID = re.compile(r"^([A-Z])N\.(\d+)\.\d+$")
_CAT_OF_LETTER = {"A": "1", "M": "8", "T": "3", "P": "5", "D": "2",
                  "O": "6", "I": "4", "C": "7"}


def note_governs_group(note_id):
    """Return (category, group) a note governs by its ID structure, or None for
    category-general (.0.) notes and IDs that don't fit the pattern."""
    m = _NOTE_ID.match(note_id)
    if not m:
        return None
    letter, num = m.group(1), m.group(2)
    if num == "0":
        return None
    cat = _CAT_OF_LETTER.get(letter)
    return (cat, f"{letter}{num}") if cat else None


def mh_groups(map_path):
    d = json.load(open(map_path, encoding="utf-8"))
    return {(g["cat"], g["grp"]) for g in d["groups"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="mbs_chunks.jsonl")
    ap.add_argument("--notes", default="mbs_notes.jsonl")
    ap.add_argument("--map", default="mbs_profession_map.json")
    ap.add_argument("--all-items", action="store_true",
                    help="attach to every item, not just mental-health groups")
    args = ap.parse_args()

    notes = [json.loads(l) for l in open(args.notes, encoding="utf-8")]
    primary, mentioned = defaultdict(list), defaultdict(list)
    group_notes = defaultdict(list)   # (cat, grp) -> [(note_id, title), ...]
    for n in notes:
        for it in n["primary_items"]:
            primary[it].append((n["note_id"], n["title"]))
        for it in n["item_refs"]:
            if it not in n["primary_items"]:
                mentioned[it].append((n["note_id"], n["title"]))
        g = note_governs_group(n["note_id"])
        if g:
            group_notes[g].append((n["note_id"], n["title"]))

    recs = [json.loads(l) for l in open(args.chunks, encoding="utf-8")]
    groups = mh_groups(args.map)
    attached = 0
    for r in recs:
        in_scope = args.all_items or (r["category"], r["group"]) in groups
        if not in_scope:
            continue
        p = primary.get(r["item_num"], [])
        m = mentioned.get(r["item_num"], [])
        grp = group_notes.get((r["category"], r["group"]), [])
        r["notes"] = {
            "primary": [{"note_id": i, "title": t} for i, t in p],
            "group": [{"note_id": i, "title": t} for i, t in grp],
            "mentioned": [{"note_id": i, "title": t} for i, t in m],
        }
        if p or m or grp:
            attached += 1

    with open(args.chunks, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scope = "all" if args.all_items else "mental-health"
    print(f"Attached notes to {attached} {scope} items (of {len(recs)} total).")


if __name__ == "__main__":
    main()
