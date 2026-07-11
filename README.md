# MBS AI Assistant

An assistant that answers questions about the Australian **Medicare Benefits
Schedule (MBS)** for mental-health practitioners — which item numbers apply,
billing conditions, eligibility context — grounded in the real schedule and its
explanatory notes.

Current coverage focuses on **mental-health items**: psychiatry (Group A8), GP
mental-health plans (A20), eating-disorder plans (A36), telehealth attendances
(A40), clinical psychology (M6), focussed psychological strategies (M7),
eating-disorder services (M16), and telehealth allied MH (M18).

> **Not billing or legal advice.** The assistant reports what descriptors and
> notes say and routes binding eligibility/interpretation to **AskMBS /
> Services Australia (13 21 50)**. The legal source is the *Health Insurance Act 1973*.

---

## Pipeline

```
MBS-XML (schedule)          Word MBS Book (explanatory notes)
      │                              │
      ▼                              ▼
 mbs_parser.py                 mbs_notes_parser.py
      │  mbs_chunks.jsonl            │  mbs_notes.jsonl
      └──────────────┬───────────────┘
                     ▼
               attach_notes.py   (joins notes → items: item-level + group-level)
                     │  mbs_chunks.jsonl (+ notes field)
                     ▼
          mbs_rag_prototype.py
   ┌─────────────────┼──────────────────┐
   ▼                 ▼                  ▼
 retrieve       query rules         answer layer
 (BM25 /        (mbs_query_          (multi-item, note-grounded,
  embeddings)    rules.py)            LLM via OpenRouter or offline fallback)
```

**Order matters:** re-running `mbs_parser.py` overwrites `mbs_chunks.jsonl` and
clears the notes field, so always re-run `attach_notes.py` afterwards.

---

## Files

### Code (tracked)
| File | Purpose |
|------|---------|
| `mbs_parser.py` | Parse the MBS XML into one retrieval-ready JSON record per item (`mbs_chunks.jsonl`). |
| `mbs_notes_parser.py` | Extract all explanatory notes from the Word MBS Book into `mbs_notes.jsonl`, with `primary_items` (title-named) and `item_refs` (all mentioned). |
| `attach_notes.py` | Join notes onto items — item-level (primary/mentioned) and group-level (via note-ID structure). Writes the `notes` field into `mbs_chunks.jsonl`. |
| `mbs_professions.py` | Group → profession scoping (which practitioner can claim an item), driven by `mbs_profession_map.json`. |
| `mbs_profession_map.json` | **First-pass, needs clinical verification.** Maps MH groups to professions; drives eligibility answers. |
| `mbs_query_rules.py` | **Curated, auditable** query-understanding layer: telehealth/collateral/initial/review/group intent rules + curated item overrides (e.g. 291 per note AN.0.30). |
| `mbs_rag_prototype.py` | The assistant: retrievers (BM25 + embeddings), rule-aware ranking, multi-item note-grounded answer layer, OpenRouter LLM seam with offline fallback. |
| `compare_retrievers.py` | Diagnostic: BM25 vs embedding retrieval side by side. |
| `requirements.txt` | `rank-bm25`, `fastembed`, `numpy` (+ `python-docx` for notes parsing). |

### Data (gitignored — regenerate locally, don't commit)
`MBS-XML-*.XML`, `Word Version - MBS Book - *.DOCX` (source inputs),
`mbs_chunks.jsonl`, `mbs_notes.jsonl`, `mbs_embeddings.npy` (generated).

---

## Setup

```bash
python3 -m pip install -r requirements.txt
```

Create `.env.local` (gitignored) from `.env.example`:

```
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=openai/gpt-4o-mini     # optional; free models exist but rate-limit
```

## Build the data

```bash
python3 mbs_parser.py MBS-XML-20260701.XML --schedule 2026-07-01 -o mbs_chunks.jsonl
python3 mbs_notes_parser.py "Word Version - MBS Book - March 2026.DOCX" --chunks mbs_chunks.jsonl -o mbs_notes.jsonl
python3 attach_notes.py --chunks mbs_chunks.jsonl --notes mbs_notes.jsonl --map mbs_profession_map.json
```

`attach_notes.py --all-items` extends the join beyond mental-health groups.

## Run

```bash
python3 mbs_rag_prototype.py        # demo queries + grounded eval (uses embeddings)
python3 compare_retrievers.py       # BM25 vs embeddings comparison
MBS_RETRIEVER=bm25 python3 mbs_rag_prototype.py   # force lexical retrieval
```

With `OPENROUTER_API_KEY` set, answers are LLM-generated (multi-item, note-grounded);
without it, the assistant falls back to a grounded, no-LLM answer template.

---

## How it works

**Retrieval.** Two interchangeable backends behind one `retrieve(query, profession, k)`
interface: `BM25Retriever` (lexical, zero-setup) and `EmbeddingRetriever`
(local `fastembed`/ONNX semantic vectors, offline after first download, cached to
`mbs_embeddings.npy`). Both apply per-item **profession scoping** and the curated
query rules.

**Query-understanding rules** (`mbs_query_rules.py`) bridge the gap between how
practitioners phrase questions and how MBS descriptors are worded. Each rule
multiplies an item's score based on a query cue and an item property:

| Rule | Effect |
|------|--------|
| telehealth | prefer face-to-face unless the query asks for video/phone |
| collateral | hide "person other than the patient" items unless asked |
| initial_new_patient | boost new-patient items for "initial/first" queries |
| initial_curated_special | modest boost + special-case label for note-only initial items (291/AN.0.30) |
| review_existing | prefer subsequent/existing-patient items for "review/follow-up" |
| individual_vs_group | demote group-therapy items unless the query says "group" |

Rules are **meant to be curated by a domain expert** — every rule carries a
`rationale` and an `mbs_note` reference.

**Notes.** The published XML has no explanatory notes, so they're recovered from
the Word MBS Book and joined two ways: **item-level** (a note names the item in
its title) and **group-level** (the note-ID encodes its group, e.g. `MN.6.x` → M6).
The answer layer injects the relevant note text so answers reflect how an item is
*intended* to be used (e.g. item 291 is a one-off assessment per AN.0.30).

**Answers.** The assistant presents the *set* of matching items with the
conditions that distinguish them (duration, new vs existing patient, age,
location, referral), cites each with its source and applicable note, and defers
binding calls to AskMBS.

---

## Known limitations / next steps

- **`mbs_profession_map.json` is a first pass** and drives eligibility answers —
  it needs verification against the Better Access rules before real use.
- **Notes join is mental-health-scoped**; run `--all-items` and validate for other
  categories to expand.
- **Some extracted note bodies are over-long** (absorb following text); the join
  uses `primary_items`/group structure so this doesn't affect associations, but
  body excerpts are truncated in prompts.
- **LLM paraphrases** descriptors/notes — grounded but interpretive; guardrails
  route binding interpretation to AskMBS.
- **Delivery surface** (web/chat UI) is not built yet — CLI only.

## Data provenance

Source data (MBS XML and the Word MBS Book) is published by the Australian
Department of Health on [MBS Online](https://www.mbsonline.gov.au). Schedule
version in use: **2026-07-01** (notes book: March 2026).
