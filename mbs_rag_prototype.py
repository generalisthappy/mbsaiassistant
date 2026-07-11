#!/usr/bin/env python3
"""
MBS assistant — runnable retrieval prototype.

Ties together the three earlier pieces:
  * the parsed chunks            (mbs_chunks.jsonl)
  * the profession scoping layer (mbs_professions.py / mbs_profession_map.json)
  * the guardrailed answer shape (mbs-assistant-guardrails.md)

What it does end to end, with NO external API:
  1. builds a BM25 index over the item chunks
  2. retrieves for a query, optionally PRE-FILTERED to a profession's groups
  3. assembles the guardrailed answer template from the retrieved item's real fields
  4. build_llm_messages() shows exactly what you'd send to a model (the only missing
     piece is the API call itself — seam marked below)
  5. evaluate() runs a small grounded eval set and reports retrieval recall@k

Swap for production: replace BM25 with vector embeddings (same retrieve() interface),
and wire call_llm() to the Anthropic API.
"""

import json
import os
import re
from rank_bm25 import BM25Okapi
from mbs_professions import groups_for_profession, professions_for_item

CHUNKS = "mbs_chunks.jsonl"
SCHEDULE_VERSION = "2026-07-01"

SYSTEM_PROMPT = (
    "You are an assistant that helps Australian health practitioners find and understand "
    "Medicare Benefits Schedule (MBS) items. Answer ONLY from the retrieved items provided "
    "below. Never state an item number, fee, or eligibility rule not present in them. Every "
    "answer must cite the item number, the schedule version, and the source link. Report "
    "what the descriptor says; route interpretation and eligibility questions to AskMBS / "
    "Services Australia (13 21 50). Give general information, not patient-specific billing "
    "advice. This is not billing or legal advice, and the Health Insurance Act 1973 is the "
    "legal source."
)

_word = re.compile(r"[a-z0-9]+")
def tokenize(s): return _word.findall((s or "").lower())


def modality(descriptor):
    d = (descriptor or "").lower()
    if d.startswith("video"):
        return "video"
    if d.startswith("phone"):
        return "phone"
    return "in_person"


_TELEHEALTH_CUE = re.compile(r"video|phone|telehealth|telephone|remote")


class MBSRetriever:
    def __init__(self, path=CHUNKS):
        self.recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        corpus = [tokenize(r["chunk_text"]) for r in self.recs]
        self.bm25 = BM25Okapi(corpus)
        # precompute per-item authoritative professions + modality
        self.item_profs = [
            set(professions_for_item(r["category"], r["group"], r["descriptor"]))
            for r in self.recs
        ]
        self.modality = [modality(r["descriptor"]) for r in self.recs]

    def retrieve(self, query, profession=None, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        # per-item profession filter (correctly handles mixed groups like A40/M18)
        if profession:
            candidate_idx = [i for i in range(len(self.recs))
                             if profession in self.item_profs[i]]
            if not candidate_idx:                       # profession not in MH map
                candidate_idx = list(range(len(self.recs)))
        else:
            candidate_idx = list(range(len(self.recs)))
        # prefer in-person unless the query asks for telehealth
        want_tele = bool(_TELEHEALTH_CUE.search(query.lower()))
        def adj(i):
            s = scores[i]
            if not want_tele and self.modality[i] != "in_person":
                s *= 0.6
            return s
        ranked = sorted(candidate_idx, key=adj, reverse=True)
        return [self.recs[i] for i in ranked[:k] if scores[i] > 0]


def assemble_answer(retrieved, profession=None):
    """Grounded answer-template output built from retrieved fields (no model needed)."""
    if not retrieved:
        return ("I don't have an MBS item in the loaded schedule that matches that. "
                "Please check MBS Online or contact AskMBS / Services Australia (13 21 50).")
    top = retrieved[0]
    profs = professions_for_item(top["category"], top["group"], top["descriptor"])
    lines = [
        f"Item(s): {top['item_num']} — {' '.join(top['descriptor'].split())[:160]}...",
        f"{top['fee_display']}",
        f"Who can claim: {', '.join(profs) if profs else 'see descriptor / notes'}",
        f"Based on: {top['schedule_version']} schedule"
        f" (effective {top['description_start_date'] or top['item_start_date']})",
        f"Source: {top['source_url']}",
        "Note: General information only, not billing advice. For binding interpretation "
        "or eligibility, contact AskMBS or Services Australia (13 21 50).",
    ]
    others = [r["item_num"] for r in retrieved[1:]]
    if others:
        lines.append(f"Related items retrieved: {', '.join(others)}")
    return "\n".join(lines)


def build_llm_messages(query, retrieved):
    """The exact payload you'd send to the model. The API call is the only missing seam."""
    context = "\n\n".join(
        f"[Item {r['item_num']} | {r['category_title']} > {r['group']} | "
        f"{r['schedule_version']}]\n{' '.join(r['descriptor'].split())}\n{r['fee_display']}\n"
        f"Source: {r['source_url']}"
        for r in retrieved
    )
    user = (f"Question: {query}\n\nRetrieved MBS items:\n{context}\n\n"
            "Answer using only these items, following the citation rules.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def call_llm(messages):
    """SEAM: drop in the Anthropic API here for production generation.
    Left unimplemented so the prototype runs without a key."""
    raise NotImplementedError(
        "Wire to the Anthropic API. assemble_answer() gives a grounded, no-LLM answer "
        "in the meantime.")


# --- evaluation ----------------------------------------------------------------
# Grounded eval: a query passes if it surfaces a correct IN-PERSON item from the
# expected (category, group) for that profession. Expected item sets are read from the
# live index at run time, so the bar reflects the real schedule, not hardcoded numbers.
EVAL_SET = [
    ("individual therapy session by a clinical psychologist", "clinical_psychologist", ("8", "M6")),
    ("focussed psychological strategies by a psychologist", "registered_psychologist", ("8", "M7")),
    ("psychiatrist attendance at least 45 minutes", "psychiatrist", ("1", "A8")),
    ("GP mental health treatment plan 20 to 40 minutes", "general_practitioner", ("1", "A20")),
    ("GP eating disorder treatment and management plan", "general_practitioner", ("1", "A36")),
    ("dietitian service for an eating disorder", "dietitian", ("8", "M16")),
]


def evaluate(retriever, k=5):
    # build expected in-person item sets per group from the index
    expected_by_group = {}
    for i, r in enumerate(retriever.recs):
        if retriever.modality[i] == "in_person":
            expected_by_group.setdefault((r["category"], r["group"]), set()).add(r["item_num"])

    hits = 0
    print(f"\nEVAL (recall@{k}, profession-filtered, in-person target):")
    for query, profession, group in EVAL_SET:
        expected = expected_by_group.get(group, set())
        got = {r["item_num"] for r in retriever.retrieve(query, profession, k)}
        ok = bool(got & expected)
        hits += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {query!r}  (target {group[1]})")
        print(f"         got {sorted(got)}")
    print(f"\n  Score: {hits}/{len(EVAL_SET)} queries surfaced a correct in-person item.")


def main():
    if not os.path.exists(CHUNKS):
        print("Run mbs_parser.py first to produce", CHUNKS); return
    r = MBSRetriever()
    print(f"Indexed {len(r.recs)} items ({SCHEDULE_VERSION}).")

    demos = [
        ("individual therapy session for my patient", "clinical_psychologist"),
        ("individual therapy session for my patient", "registered_psychologist"),
        ("initial psychiatric assessment", "psychiatrist"),
    ]
    for q, prof in demos:
        print("\n" + "=" * 78)
        print(f"Q: {q!r}   (as: {prof})")
        hits = r.retrieve(q, prof, k=4)
        print("-" * 78)
        print(assemble_answer(hits, prof))

    evaluate(r, k=5)


if __name__ == "__main__":
    main()
