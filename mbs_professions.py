#!/usr/bin/env python3
"""
Group -> profession scoping for MBS items.

Eligibility is driven by the GROUP (authoritative, from mbs_profession_map.json), not by
descriptor scraping. For "mixed" groups — telehealth bundles where one group holds items
for several provider types — we refine per item by looking for the specific provider named
in that item's descriptor, falling back to the group's provider set if none is found.

Use groups_for_profession() to pre-filter the vector index before search, and
professions_for_item() to tag records at ingest time.
"""

import json
import re
import os

_MAP_PATH = os.path.join(os.path.dirname(__file__), "mbs_profession_map.json")

# Descriptor phrase -> profession, used only to refine MIXED groups per item.
_REFINE = [
    ("psychiatrist", r"psychiatr"),
    ("clinical_psychologist", r"clinical psychologist"),
    ("registered_psychologist", r"(eligible |registered )psychologist|\bpsychologist"),
    ("general_practitioner", r"general practitioner"),
    ("consultant_physician", r"consultant physician"),
    ("specialist", r"\bspecialist\b"),
    ("dietitian", r"dietitian"),
    ("fps_allied_health", r"occupational therap|social worker|mental health nurse"),
]


def load_map(path=_MAP_PATH):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    idx = {(g["cat"], g["grp"]): g for g in data["groups"]}
    return data, idx


_DATA, _IDX = load_map()


def groups_for_profession(profession):
    """Return [(category, group), ...] a profession can claim in. Use to pre-filter."""
    return [(g["cat"], g["grp"]) for g in _DATA["groups"]
            if profession in g["professions"]]


def _refine_from_descriptor(descriptor, allowed):
    d = re.sub(r"other than [^)\-—]*", " ", (descriptor or "").lower())
    # "clinical psychologist" must not also count as a (generic) registered psychologist,
    # so test the generic-psychologist pattern against a copy with clinical stripped out.
    d_no_clinical = d.replace("clinical psychologist", " ")
    found = []
    for prof, pat in _REFINE:
        target = d_no_clinical if prof == "registered_psychologist" else d
        if re.search(pat, target):
            found.append(prof)
    return [p for p in found if p in allowed]


def professions_for_item(category, group, descriptor=""):
    """Authoritative professions for an item. Group-driven; refined for mixed groups."""
    g = _IDX.get((category, group))
    if not g:
        return []              # group not in the mental-health map
    if not g.get("mixed"):
        return list(g["professions"])
    refined = _refine_from_descriptor(descriptor, set(g["professions"]))
    return refined or list(g["professions"])   # fall back to group set if unclear


if __name__ == "__main__":
    # Demo against the parsed chunks, if present.
    import sys
    chunks = "mbs_chunks.jsonl"
    if not os.path.exists(chunks):
        print("Run mbs_parser.py first to produce", chunks); sys.exit(0)
    recs = [json.loads(l) for l in open(chunks)]

    print("Groups a CLINICAL PSYCHOLOGIST can claim in:",
          groups_for_profession("clinical_psychologist"))
    print("Groups a PSYCHIATRIST can claim in:",
          groups_for_profession("psychiatrist"), "\n")

    # Tag every record and count by profession
    from collections import Counter
    c = Counter()
    mixed_examples = []
    for r in recs:
        profs = professions_for_item(r["category"], r["group"], r["descriptor"])
        r["professions"] = profs
        for p in profs:
            c[p] += 1
        if r["group"] == "M18" and profs and len(mixed_examples) < 3:
            mixed_examples.append((r["item_num"], profs))

    print("Item counts by profession (group-scoped):")
    for p, n in c.most_common():
        print(f"  {p}: {n}")

    print("\nRefinement working inside mixed group M18 (telehealth):")
    for num, profs in mixed_examples:
        print(f"  item {num} -> {profs}")
