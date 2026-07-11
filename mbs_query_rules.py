#!/usr/bin/env python3
"""
MBS query-understanding rules — a CURATED, auditable layer.

Practitioners and MBS descriptors use different words for the same thing
("initial" vs "new patient", "session" vs "attendance/health service"). Neither
keyword nor embedding retrieval bridges that reliably, so this module applies a
small set of hand-curated rules that nudge retrieval scores based on (a) a signal
in the QUERY and (b) a property of the ITEM.

This file is meant to be READ AND EXTENDED BY A DOMAIN EXPERT. Every rule carries
a rationale and, where the behaviour comes from an MBS note rather than the
descriptor text, the governing note reference. Editing factors, cues, or the
curated item lists below does not require touching the retrieval code.

Each rule multiplies an item's base relevance score:
  name       - short identifier
  query_cue  - regex; if it matches the query, the rule is "active"
  item_test  - fn(record) -> bool; which items the rule targets
  active     - multiplier for matching items when the cue IS present
  inactive   - multiplier for matching items when the cue is ABSENT
  rationale  - why the rule exists (for audit)
  mbs_note   - governing MBS note, if the behaviour isn't in the descriptor text

NOTE: The published MBS XML does NOT contain the explanatory notes (category /
group / item notes such as AN.0.30). Any behaviour that depends on a note must be
encoded here as a curated override until the notes source is joined into the data.
"""

import re

# ---------------------------------------------------------------------------
# Curated item-level overrides — domain knowledge NOT derivable from descriptors
# ---------------------------------------------------------------------------
# Items valid for an INITIAL / first appointment even though the descriptor never
# says "new patient". Item number -> reason (cite the governing note).
INITIAL_ELIGIBLE_ITEMS = {
    "291": "AN.0.30 — one-off comprehensive psychiatrist assessment; may be used as "
           "a first appointment and can be combined with other items. Not worded as "
           "'new patient' in the descriptor.",
}

# ---------------------------------------------------------------------------
# Item-property detectors (descriptor-derived unless noted)
# ---------------------------------------------------------------------------
_VIDEO = re.compile(r"\bvideo attendance\b|attendance is by video|\bby video\b|video conference")
_PHONE = re.compile(r"\bphone attendance\b|attendance is by (?:phone|telephone)|\bby (?:phone|telephone)\b")
_COLLATERAL = re.compile(r"person other than the patient|patient is not in attendance")
_GROUP = re.compile(r"as part of a group|group of \d+")
_SUBSEQUENT = re.compile(r"after the first|after the initial attendance|each attendance after|"
                         r"subsequent attendance|being managed by")


def modality(descriptor):
    """in_person / video / phone — telehealth flagged wherever it appears."""
    d = (descriptor or "").lower()
    if d.startswith("video") or _VIDEO.search(d):
        return "video"
    if d.startswith("phone") or _PHONE.search(d):
        return "phone"
    return "in_person"


def is_collateral(descriptor):
    return bool(_COLLATERAL.search((descriptor or "").lower()))


def is_group(descriptor):
    return bool(_GROUP.search((descriptor or "").lower()))


def is_text_new_patient(r):
    """Standard new-patient item — the descriptor itself says 'new patient'."""
    return "new patient" in (r.get("descriptor") or "").lower()


def is_curated_initial(r):
    """Initial-eligible ONLY via an explanatory note (special one-off, e.g. 291 per
    AN.0.30) — the descriptor does not say 'new patient'."""
    return (str(r.get("item_num")) in INITIAL_ELIGIBLE_ITEMS
            and not is_text_new_patient(r))


def is_initial_item(r):
    """Any initial-eligible item (standard new-patient OR curated special case).
    Used to exclude these from 'subsequent' and to label them."""
    return is_text_new_patient(r) or str(r.get("item_num")) in INITIAL_ELIGIBLE_ITEMS


def is_subsequent_item(r):
    """Subsequent / existing-patient attendance. Excludes anything we've marked as
    initial-eligible so a one-off assessment (e.g. 291) isn't treated as a review."""
    if is_initial_item(r):
        return False
    return bool(_SUBSEQUENT.search((r.get("descriptor") or "").lower()))


# ---------------------------------------------------------------------------
# Query cues
# ---------------------------------------------------------------------------
_TELEHEALTH_CUE = re.compile(r"video|phone|telehealth|telephone|remote")
_COLLATERAL_CUE = re.compile(r"collateral|family|carer|relative|parent|partner|friend|"
                             r"person other than|not in attendance|third[- ]party")
_INITIAL_CUE = re.compile(r"\b(initial|first|new patient|new referral|intake|new consult)\b")
_REVIEW_CUE = re.compile(r"\b(review|subsequent|follow[- ]?up|ongoing|existing patient|continuing)\b")
_GROUP_CUE = re.compile(r"\bgroup\b")


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------
RULES = [
    {
        "name": "telehealth",
        "query_cue": _TELEHEALTH_CUE,
        "item_test": lambda r: modality(r.get("descriptor")) != "in_person",
        "active": 1.0, "inactive": 0.6,
        "rationale": "Prefer face-to-face by default; only surface video/phone items "
                     "when the query asks for telehealth.",
        "mbs_note": None,
    },
    {
        "name": "collateral",
        "query_cue": _COLLATERAL_CUE,
        "item_test": lambda r: is_collateral(r.get("descriptor")),
        "active": 1.0, "inactive": 0.15,
        "rationale": "Items for interviewing someone other than the patient (family "
                     "collateral history) are irrelevant to patient-facing queries "
                     "unless explicitly requested.",
        "mbs_note": None,
    },
    {
        "name": "initial_new_patient",
        "query_cue": _INITIAL_CUE,
        "item_test": is_text_new_patient,
        "active": 2.0, "inactive": 1.0,
        "rationale": "'Initial/first' consults map to MBS 'new patient' items that "
                     "practitioner wording ('assessment') doesn't lexically match.",
        "mbs_note": None,
    },
    {
        "name": "initial_curated_special",
        "query_cue": _INITIAL_CUE,
        "item_test": is_curated_initial,
        "active": 1.2, "inactive": 1.0,
        "rationale": "Items initial-eligible only via an explanatory note (special "
                     "one-off cases like 291 per AN.0.30) get a modest boost — enough "
                     "to surface, but not to outrank the standard new-patient items.",
        "mbs_note": "AN.0.30",
    },
    {
        "name": "review_existing",
        "query_cue": _REVIEW_CUE,
        "item_test": is_subsequent_item,
        "active": 1.8, "inactive": 1.0,
        "rationale": "'Review/subsequent/follow-up' queries should prefer subsequent / "
                     "existing-patient attendance items over new-patient ones.",
        "mbs_note": None,
    },
    {
        "name": "individual_vs_group",
        "query_cue": _GROUP_CUE,
        "item_test": lambda r: is_group(r.get("descriptor")),
        "active": 1.5, "inactive": 0.5,
        "rationale": "Group-therapy items should surface only when the query mentions "
                     "a group; individual/standard queries should demote them.",
        "mbs_note": None,
    },
]


def active_rules(query):
    """Return {rule_name: bool} for whether each rule's cue fires on the query."""
    ql = (query or "").lower()
    return {rule["name"]: bool(rule["query_cue"].search(ql)) for rule in RULES}
