#!/usr/bin/env python3
"""
Extract MBS explanatory notes from the Word book into structured JSONL.

The published MBS XML carries item descriptors and fees but NOT the explanatory
notes (category / group / item notes such as AN.0.30). Those live in the Word
"MBS Book". This parser recovers them.

Segmentation: in the .docx, every note begins with a paragraph whose text starts
with a note-ID (e.g. "AN.0.30 ..."); the following paragraphs are its body until
the next note-ID header. Each note becomes:

    {note_id, prefix, title, body, item_refs}

item_refs = the real MBS item numbers mentioned in the note, intersected with the
schedule so years/dollar amounts/paragraph numbers aren't mistaken for items.

Usage:
    python mbs_notes_parser.py "Word Version - MBS Book - March 2026.DOCX" \
        --chunks mbs_chunks.jsonl -o mbs_notes.jsonl
"""

import argparse
import json
import re
from docx import Document

ID_RE = re.compile(r"^([A-Z]{1,3}\.\d+\.\d+)\b")
NUM_RE = re.compile(r"\b(\d{1,5})\b")


def load_item_ids(chunks_path):
    return {json.loads(l)["item_num"] for l in open(chunks_path, encoding="utf-8")}


def parse_notes(docx_path, valid_items):
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs]
    notes, cur = [], None
    for t in paras:
        m = ID_RE.match(t)
        if m:
            if cur:
                notes.append(cur)
            nid = m.group(1)
            cur = {"note_id": nid, "prefix": nid.split(".")[0],
                   "title": t[len(nid):].strip(" -–—:"), "_body": []}
        elif cur is not None and t:
            cur["_body"].append(t)
    if cur:
        notes.append(cur)

    def items_in(text):
        return sorted({x for x in NUM_RE.findall(text) if x in valid_items},
                      key=lambda s: (len(s), s))

    for n in notes:
        body = " ".join(n.pop("_body"))
        n["body"] = body
        # primary_items: named in the note's title/header — the items the note is
        # ABOUT (authoritative link). item_refs: every schedule item mentioned
        # anywhere (title + body), incl. cross-references.
        n["primary_items"] = items_in(n["title"])
        n["item_refs"] = items_in(f"{n['title']} {body}")
    return notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("docx")
    ap.add_argument("--chunks", default="mbs_chunks.jsonl")
    ap.add_argument("-o", "--out", default="mbs_notes.jsonl")
    args = ap.parse_args()

    valid = load_item_ids(args.chunks)
    notes = parse_notes(args.docx, valid)
    with open(args.out, "w", encoding="utf-8") as f:
        for n in notes:
            f.write(json.dumps(n, ensure_ascii=False) + "\n")
    with_refs = sum(1 for n in notes if n["item_refs"])
    print(f"Wrote {len(notes)} notes to {args.out} "
          f"({with_refs} reference at least one schedule item).")


if __name__ == "__main__":
    main()
