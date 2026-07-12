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
    ("fps_allied_health", r"occupational therap|social worker"),
    ("nurse_practitioner", r"nurse practitioner"),
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


def _in_scope(scope, subgroup, descriptor):
    """True if an item belongs in a scoped group's mental-health subset.

    Hybrid test: passes if the item's subgroup is whitelisted OR its descriptor matches
    the include regex. Subgroups anchor the structurally-clean slice (e.g. psychiatry
    telehealth); the regex catches items interleaved in general subgroups (e.g. GP
    mental-health telehealth)."""
    subs = scope.get("subgroups")
    if subs and subgroup is not None and str(subgroup) in set(subs):
        return True
    inc = scope.get("descriptor_include")
    if inc and re.search(inc, (descriptor or "").lower()):
        return True
    return False


def professions_for_item(category, group, descriptor="", subgroup=None):
    """Authoritative professions for an item. Group-driven; refined for mixed groups.

    A group may carry a "scope" block (e.g. A40, the general telehealth group) that
    restricts which of its items belong in the mental-health map, by subgroup whitelist
    and/or a descriptor regex. Items outside the scope return [] (not in the MH map)."""
    g = _IDX.get((category, group))
    if not g:
        return []              # group not in the mental-health map
    scope = g.get("scope")
    if scope and not _in_scope(scope, subgroup, descriptor):
        return []              # in the group, but outside its MH-relevant scope
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
        profs = professions_for_item(r["category"], r["group"], r["descriptor"], r.get("subgroup"))
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
