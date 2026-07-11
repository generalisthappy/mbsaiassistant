#!/usr/bin/env python3
"""
MBS XML -> retrieval-ready chunks.

Parses a Medicare Benefits Schedule XML file (as published on MBS Online) into one
JSON record per item, shaped for a RAG index. Each record carries an embed-ready text
blob plus the metadata the assistant's guardrails depend on (item number, category/group
path, effective dates, fee, provider hints, source URL).

Usage:
    python mbs_parser.py MBS-XML-20260701.XML --schedule 2026-07-01 -o mbs_chunks.jsonl

Notes / known limits of the source file:
  * The XML carries item DESCRIPTORS and fees, but NOT the explanatory notes
    (category/group/item notes). Those are published separately and MUST be joined in
    for interpretation questions — see attach_notes() for the seam.
  * The <ProviderType> field is empty on every item in practice, so provider eligibility
    is inferred heuristically from the descriptor and group. Treat provider_hints as a
    starting signal to be reinforced with the actual program rules (e.g. Better Access
    for psychologists), not as authoritative.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET

# --- reference data -------------------------------------------------------------

# Seed only the category titles we're confident are stable. Anything not listed falls
# back to "Category N". For production, prefer pulling the authoritative category/group
# titles from the schedule's own structure/notes rather than hardcoding.
CATEGORY_TITLES = {
    "1": "Professional Attendances",
    "2": "Diagnostic Procedures and Investigations",
    "3": "Therapeutic Procedures",
    "4": "Oral and Maxillofacial Services",
    "5": "Diagnostic Imaging Services",
    "6": "Pathology Services",
    "7": "Cleft Lip and Cleft Palate Services",
    "8": "Miscellaneous Services",
}

# Heuristic descriptor phrases -> provider tag. First-pass signal only.
PROVIDER_PATTERNS = [
    ("psychiatrist", r"psychiatr"),
    ("consultant_physician", r"consultant physician"),
    ("general_practitioner", r"general practitioner"),
    ("specialist", r"\bspecialist\b"),
    ("clinical_psychologist", r"clinical psychologist"),
    ("psychologist", r"\bpsychologist"),
    ("nurse_practitioner", r"nurse practitioner"),
    ("midwife", r"midwife|midwifery"),
    ("optometrist", r"optometrist"),
    ("allied_health", r"allied health|occupational therap|physiotherap|"
                      r"speech patholog|dietitian|osteopath|chiropract"),
]

MBS_ITEM_URL = "https://www9.health.gov.au/mbs/search.cfm?q={item}"


# --- helpers --------------------------------------------------------------------

def iso_date(ddmmyyyy: str):
    """Convert MBS 'DD.MM.YYYY' to ISO 'YYYY-MM-DD'. Returns None if blank/malformed."""
    s = (ddmmyyyy or "").strip()
    m = re.match(r"^(\d{2})\.(\d{2})\.(\d{4})$", s)
    if not m:
        return None
    d, mo, y = m.groups()
    return f"{y}-{mo}-{d}"


def txt(el, tag):
    v = el.findtext(tag)
    return (v or "").strip()


def clean_descriptor(s: str) -> str:
    """Tidy whitespace; the source packs list items as '(a)...(b)...' with no breaks."""
    s = (s or "").strip()
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def infer_provider_hints(descriptor: str):
    # Drop exclusion clauses ("other than psychiatry", "(other than a general
    # practitioner)") so we don't tag a provider the item explicitly excludes.
    # This is a heuristic guard, not a substitute for the program-rules layer.
    d = re.sub(r"other than [^)\-—]*", " ", descriptor.lower())
    return [tag for tag, pat in PROVIDER_PATTERNS if re.search(pat, d)]


def fee_display(rec):
    """Human-readable fee line, handling normal vs derived-fee items."""
    if rec["fee_type"] == "D" and rec["derived_fee"]:
        return f"Derived fee: {rec['derived_fee']}"
    if rec["schedule_fee"]:
        parts = [f"Schedule fee: ${rec['schedule_fee']}"]
        benefits = []
        for label, key in (("100%", "benefit_100"), ("85%", "benefit_85"),
                           ("75%", "benefit_75")):
            if rec.get(key):
                benefits.append(f"{label} = ${rec[key]}")
        if benefits:
            parts.append("Benefit: " + ", ".join(benefits))
        return "; ".join(parts)
    return "Fee: not specified in item data"


# --- core -----------------------------------------------------------------------

def parse_item(el, schedule_version):
    descriptor = clean_descriptor(txt(el, "Description"))
    category = txt(el, "Category")
    group = txt(el, "Group")
    subgroup = txt(el, "SubGroup")

    rec = {
        "item_num": txt(el, "ItemNum"),
        "sub_item_num": txt(el, "SubItemNum"),
        "schedule_version": schedule_version,
        "category": category,
        "category_title": CATEGORY_TITLES.get(category, f"Category {category}"),
        "group": group,
        "subgroup": subgroup,
        "subheading": txt(el, "SubHeading"),
        "item_type": txt(el, "ItemType"),
        "fee_type": txt(el, "FeeType"),
        "benefit_type": txt(el, "BenefitType"),
        "schedule_fee": txt(el, "ScheduleFee"),
        "benefit_100": txt(el, "Benefit100"),
        "benefit_85": txt(el, "Benefit85"),
        "benefit_75": txt(el, "Benefit75"),
        "derived_fee": txt(el, "DerivedFee"),
        "item_start_date": iso_date(txt(el, "ItemStartDate")),
        "item_end_date": iso_date(txt(el, "ItemEndDate")),
        "fee_start_date": iso_date(txt(el, "FeeStartDate")),
        "description_start_date": iso_date(txt(el, "DescriptionStartDate")),
        "descriptor": descriptor,
        # notes are joined from a separate source; empty until attach_notes() runs
        "notes": "",
        "source_url": MBS_ITEM_URL.format(item=txt(el, "ItemNum")),
    }
    rec["is_active"] = rec["item_end_date"] is None
    rec["provider_hints"] = infer_provider_hints(descriptor)
    rec["fee_display"] = fee_display(rec)
    return rec


def build_chunk_text(rec):
    """The text that gets embedded. Front-load the identifiers retrieval keys on."""
    path = " > ".join(p for p in [rec["category_title"], rec["group"],
                                  rec["subgroup"], rec["subheading"]] if p)
    lines = [
        f"MBS item {rec['item_num']} ({rec['schedule_version']} schedule)",
        f"Classification: {path}",
        f"Descriptor: {rec['descriptor']}",
        rec["fee_display"],
    ]
    if rec["provider_hints"]:
        lines.append("Provider (inferred): " + ", ".join(rec["provider_hints"]))
    if rec["notes"]:
        lines.append(f"Associated notes: {rec['notes']}")
    lines.append(f"Effective from: {rec['description_start_date'] or rec['item_start_date']}")
    return "\n".join(lines)


def attach_notes(rec, notes_index):
    """Join point for explanatory notes. notes_index: {item_num: notes_text}.
    Wire this to your notes source (create-publication output, item pages, or a
    notes XML). Left as a no-op-friendly stub so the pipeline runs without notes."""
    if notes_index:
        rec["notes"] = notes_index.get(rec["item_num"], "")
    return rec


def parse_file(path, schedule_version, notes_index=None):
    tree = ET.parse(path)
    for el in tree.getroot().findall("Data"):
        rec = parse_item(el, schedule_version)
        attach_notes(rec, notes_index)
        rec["chunk_text"] = build_chunk_text(rec)
        yield rec


def main():
    ap = argparse.ArgumentParser(description="Parse MBS XML into RAG chunks (JSONL).")
    ap.add_argument("xml_path")
    ap.add_argument("--schedule", required=True,
                    help="Schedule version label, e.g. 2026-07-01")
    ap.add_argument("-o", "--output", default="mbs_chunks.jsonl")
    ap.add_argument("--active-only", action="store_true",
                    help="Emit only items with no end date")
    ap.add_argument("--limit", type=int, default=0, help="Cap records (for sampling)")
    args = ap.parse_args()

    n = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for rec in parse_file(args.xml_path, args.schedule):
            if args.active_only and not rec["is_active"]:
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    print(f"Wrote {n} records to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
