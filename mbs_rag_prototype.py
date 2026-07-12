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
  4. build_llm_messages() shows exactly what you'd send to a model
  5. call_llm() generates via OpenRouter (OpenAI-compatible); answer() orchestrates
     retrieval + generation and falls back to the grounded template with no key
  6. evaluate() runs a small grounded eval set and reports retrieval recall@k

Generation: set OPENROUTER_API_KEY to enable the LLM. Model defaults to a free
open-source model (DEFAULT_MODEL) and is overridable via OPENROUTER_MODEL. With no
key, everything still runs — answer() returns the grounded, no-LLM template.

Swap for production: replace BM25 with vector embeddings (same retrieve() interface).
"""

import json
import os
import re
from rank_bm25 import BM25Okapi
from mbs_professions import groups_for_profession, professions_for_item
from mbs_not_billable import not_billable_hit, explain
from mbs_query_rules import (RULES, active_rules, modality, is_collateral,
                             is_group, is_initial_item, is_subsequent_item,
                             INITIAL_ELIGIBLE_ITEMS)

CHUNKS = "mbs_chunks.jsonl"
SCHEDULE_VERSION = "2026-07-01"

# OpenRouter (OpenAI-compatible) generation config.
# Model is overridable via OPENROUTER_MODEL so it can change without a code edit.
# Default is a cheap paid model that's reliably available; the free open-source
# option (meta-llama/llama-3.3-70b-instruct:free) works too but is frequently
# rate-limited upstream, so it isn't the default.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"

# Local semantic-embedding retrieval (fastembed / ONNX, offline after first run).
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_CACHE = "mbs_embeddings.npy"

SYSTEM_PROMPT = (
    "You are an assistant that helps Australian health practitioners find and understand "
    "Medicare Benefits Schedule (MBS) items. Answer ONLY from the retrieved items provided "
    "below. Never state an item number, fee, or eligibility rule not present in them. Every "
    "answer must cite the item number, the schedule version, and the source link. Report "
    "what the descriptor says; route interpretation and eligibility questions to AskMBS / "
    "Services Australia (13 21 50). Give general information, not patient-specific billing "
    "advice. This is not billing or legal advice, and the Health Insurance Act 1973 is the "
    "legal source. When several retrieved items could apply, present them side by side with "
    "the conditions that distinguish them (duration, new vs existing patient, patient age "
    "limits, location, referral requirements) rather than asserting a single item."
)

_word = re.compile(r"[a-z0-9]+")
def tokenize(s): return _word.findall((s or "").lower())


# Query-understanding rules (telehealth / collateral / initial / review / group)
# and the item-property detectors (modality, is_collateral, ...) live in the
# curated, auditable mbs_query_rules module and are imported at the top.

_DUR = re.compile(r"(at least|more than|not more than|less than)\s+(\d+)\s+minutes")


def item_conditions(r):
    """Extract the high-signal conditions that distinguish similar items, for a
    compact one-line summary in the grounded (no-LLM) answer."""
    d = " ".join((r.get("descriptor") or "").split())
    dl = d.lower()
    bits = []
    durs = _DUR.findall(dl)
    if durs:
        bits.append(" / ".join(f"{phrase} {n} min" for phrase, n in durs[:2]))
    if "new patient" in dl:
        bits.append("new patient / initial")
    elif str(r.get("item_num")) in INITIAL_ELIGIBLE_ITEMS:
        bits.append("initial — special one-off (see note)")
    if is_group(r.get("descriptor")):
        bits.append("group therapy")
    age = re.search(r"aged under (\d+)|under (\d+) years|aged (\d+) years or", dl)
    if age:
        n = next(g for g in age.groups() if g)
        bits.append(f"age ~{n}")
    if "consulting rooms" in dl and "hospital" in dl:
        bits.append("rooms/hospital")
    elif "hospital" in dl:
        bits.append("hospital")
    elif "consulting rooms" in dl:
        bits.append("consulting rooms")
    mod = modality(r.get("descriptor"))
    if mod != "in_person":
        bits.append(mod)
    if is_collateral(r.get("descriptor")):
        bits.append("collateral (non-patient)")
    if "following referral" in dl or "after referral" in dl or "referred" in dl:
        bits.append("referral req'd")
    return " · ".join(bits)


class BaseRetriever:
    """Shared record loading, profession scoping, and modality preference.

    Subclasses implement _scores(query) -> per-record relevance array and
    _keep(score) -> whether a scored record clears the relevance floor. The
    retrieve(query, profession, k) contract is identical across backends so
    they are drop-in interchangeable.
    """

    def __init__(self, path=CHUNKS):
        self.recs = [json.loads(l) for l in open(path, encoding="utf-8")]
        # precompute per-item authoritative professions + modality
        self.item_profs = [
            set(professions_for_item(r["category"], r["group"], r["descriptor"], r.get("subgroup")))
            for r in self.recs
        ]
        self.modality = [modality(r["descriptor"]) for r in self.recs]
        # Precompute each curated rule's item mask once, for fast query-time scoring.
        self._rule_masks = {
            rule["name"]: [rule["item_test"](r) for r in self.recs]
            for rule in RULES
        }

    def _scores(self, query):
        raise NotImplementedError

    def _keep(self, score):
        return True

    def retrieve(self, query, profession=None, k=5):
        scores = self._scores(query)
        # per-item profession filter (correctly handles mixed groups like A40/M18)
        if profession:
            candidate_idx = [i for i in range(len(self.recs))
                             if profession in self.item_profs[i]]
            if not candidate_idx:                       # profession not in MH map
                candidate_idx = list(range(len(self.recs)))
        else:
            candidate_idx = list(range(len(self.recs)))
        # Apply the curated query-understanding rules: each rule multiplies an
        # item's score depending on whether its cue fired on this query.
        active = active_rules(query)
        def adj(i):
            s = scores[i]
            for rule in RULES:
                if self._rule_masks[rule["name"]][i]:
                    s *= rule["active"] if active[rule["name"]] else rule["inactive"]
            return s
        ranked = sorted(candidate_idx, key=adj, reverse=True)
        return [self.recs[i] for i in ranked[:k] if self._keep(scores[i])]


class BM25Retriever(BaseRetriever):
    """Lexical BM25 retrieval. Fast, no dependencies beyond rank-bm25, but
    keyword-only: it can surface lexically similar yet clinically wrong items."""

    def __init__(self, path=CHUNKS):
        super().__init__(path)
        corpus = [tokenize(r["chunk_text"]) for r in self.recs]
        self.bm25 = BM25Okapi(corpus)

    def _scores(self, query):
        return self.bm25.get_scores(tokenize(query))

    def _keep(self, score):
        return score > 0


# Backward-compatible alias: existing callers get BM25 by default.
MBSRetriever = BM25Retriever


class EmbeddingRetriever(BaseRetriever):
    """Semantic retrieval via local sentence embeddings (fastembed / ONNX).

    Runs fully offline after the first model download — no API key, no per-query
    cost. Document embeddings are cached to disk (keyed to model + schedule
    version + record count) so startup is instant on subsequent runs.
    """

    def __init__(self, path=CHUNKS, model=EMBED_MODEL, cache=EMBED_CACHE, min_score=0.30):
        super().__init__(path)
        import numpy as np
        from fastembed import TextEmbedding
        self.np = np
        self.min_score = min_score
        self.embedder = TextEmbedding(model)
        self.doc_emb = self._load_or_build(model, cache)

    def _doc_text(self, r):
        # Embed the group path + descriptor; the descriptor carries the clinical
        # meaning that lexical matching misses. Truncate the long procedural tail
        # of some descriptors — the distinguishing clinical terms are up front, and
        # it keeps the one-time build fast.
        desc = ' '.join(r['descriptor'].split())[:512]
        return f"{r['category_title']} {r['group']}: {desc}"

    def _embed_norm(self, texts):
        np = self.np
        emb = np.asarray(list(self.embedder.embed(list(texts))), dtype="float32")
        # Sanitise: drop any non-finite values, then L2-normalise with a safe
        # divisor so all-zero / degenerate embeddings don't produce nan/inf.
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def _load_or_build(self, model, cache):
        np = self.np
        # 'v2' invalidates any cache built with the earlier normalisation bug.
        sig = f"v2|{model}|{self.recs[0]['schedule_version']}|{len(self.recs)}"
        meta = cache + ".meta"
        if os.path.exists(cache) and os.path.exists(meta) and open(meta).read() == sig:
            return np.load(cache)
        emb = self._embed_norm(self._doc_text(r) for r in self.recs)
        np.save(cache, emb)
        with open(meta, "w") as f:
            f.write(sig)
        return emb

    def _scores(self, query):
        np = self.np
        q = self._embed_norm([query])[0]
        # numpy's float32 matmul can raise spurious FP-state warnings on some
        # BLAS backends (e.g. macOS Accelerate); guard it and sanitise output so
        # any degenerate row sorts to the bottom rather than as nan.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            scores = self.doc_emb @ q     # cosine similarity (both L2-normalised)
        return np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

    def _keep(self, score):
        return score >= self.min_score


def make_retriever(kind=None, path=CHUNKS):
    """Choose a retriever. Defaults to embeddings (better precision); set
    MBS_RETRIEVER=bm25 to force lexical, or falls back to BM25 automatically if
    the embedding backend isn't installed."""
    kind = kind or os.environ.get("MBS_RETRIEVER", "embedding")
    if kind == "bm25":
        return BM25Retriever(path)
    try:
        return EmbeddingRetriever(path)
    except Exception as e:
        print(f"  [warn] embedding retriever unavailable ({type(e).__name__}: {e}); "
              "using BM25")
        return BM25Retriever(path)


def assemble_answer(retrieved, profession=None, max_items=4):
    """Grounded, multi-item answer built from retrieved fields (no model needed).

    Presents up to max_items candidates side by side with the conditions that
    distinguish them, so an ambiguous query returns a useful comparison rather
    than one (possibly wrong) asserted item.
    """
    if not retrieved:
        return ("I don't have an MBS item in the loaded schedule that matches that. "
                "Please check MBS Online or contact AskMBS / Services Australia (13 21 50).")

    items = retrieved[:max_items]
    top = items[0]
    profs = professions_for_item(top["category"], top["group"], top["descriptor"], top.get("subgroup"))

    lines = [
        f"Possible MBS items (schedule {top['schedule_version']}). More than one may "
        "apply — check the distinguishing conditions against your situation:",
        "",
    ]
    for r in items:
        cond = item_conditions(r)
        head = f"• Item {r['item_num']} ({r['category_title']} > {r['group']})"
        if cond:
            head += f" — {cond}"
        lines.append(head)
        lines.append(f"    {' '.join(r['descriptor'].split())[:200]}...")
        lines.append(f"    {r['fee_display']}   Source: {r['source_url']}")
        prim = (r.get("notes") or {}).get("primary", [])
        if prim:
            lines.append("    Applicable note(s): "
                         + "; ".join(f"{p['note_id']} {p['title'][:50]}" for p in prim))
    lines += [
        "",
        f"Who can claim (item {top['item_num']}): "
        f"{', '.join(profs) if profs else 'see descriptor / notes'}.",
        "General information only, not billing advice. Eligibility and binding "
        "interpretation: AskMBS or Services Australia (13 21 50). Legal source: "
        "Health Insurance Act 1973.",
    ]
    return "\n".join(lines)


_NOTES_INDEX = None


def load_notes_index(path="mbs_notes.jsonl"):
    """Lazy-load {note_id: note_record} from the extracted notes file, if present."""
    global _NOTES_INDEX
    if _NOTES_INDEX is None:
        _NOTES_INDEX = {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                n = json.loads(line)
                _NOTES_INDEX[n["note_id"]] = n
    return _NOTES_INDEX


def _note_excerpt(body, item_num, body_chars=700):
    """Return a body excerpt of at most ~body_chars. If the item number appears in the
    body, centre the window on its first occurrence so the clause that actually governs
    the item is included (many notes are long and mention the item deep in the text);
    otherwise fall back to the head of the body."""
    body = " ".join((body or "").split())
    if len(body) <= body_chars:
        return body
    pos = -1
    if item_num:
        m = re.search(r"\b" + re.escape(str(item_num)) + r"\b", body)
        if m:
            pos = m.start()
    if pos == -1:
        return body[:body_chars].rstrip() + " …"
    half = body_chars // 2
    end = min(len(body), pos + half)
    start = max(0, end - body_chars)
    end = min(len(body), start + body_chars)
    snip = body[start:end].strip()
    if start > 0:
        snip = "… " + snip
    if end < len(body):
        snip = snip + " …"
    return snip


def primary_notes_for(retrieved, max_notes=4, body_chars=700):
    """Deduped explanatory notes for the retrieved items, most-specific first
    (item-title 'primary' notes, then group-level notes): [(id, title, excerpt)].
    Each excerpt is centred on the surfacing item's number when present in the body."""
    idx = load_notes_index()
    seen, out = set(), []
    for kind in ("primary", "group"):
        for r in retrieved:
            for ref in (r.get("notes") or {}).get(kind, []):
                nid = ref["note_id"]
                if nid in seen:
                    continue
                seen.add(nid)
                n = idx.get(nid)
                if n:
                    out.append((nid, ref["title"],
                                _note_excerpt(n["body"], r.get("item_num"), body_chars)))
                if len(out) >= max_notes:
                    return out
    return out


def build_llm_messages(query, retrieved):
    """The exact payload you'd send to the model. The API call is the only missing seam."""
    context = "\n\n".join(
        f"[Item {r['item_num']} | {r['category_title']} > {r['group']} | "
        f"{r['schedule_version']}]\n{' '.join(r['descriptor'].split())}\n{r['fee_display']}\n"
        f"Source: {r['source_url']}"
        for r in retrieved
    )
    notes = primary_notes_for(retrieved)
    notes_block = ""
    if notes:
        notes_block = "\n\nApplicable explanatory notes (use for context only):\n" + \
            "\n\n".join(f"[{nid}] {title}\n{body}" for nid, title, body in notes)
    user = (
        f"Question: {query}\n\nCandidate MBS items:\n{context}{notes_block}\n\n"
        "Using ONLY the items and notes above:\n"
        "- If several plausibly apply, list each with the conditions that distinguish it "
        "(duration, new vs existing patient, patient age limits, location — consulting "
        "rooms / hospital / telehealth, referral requirements) so the practitioner can "
        "identify the right one.\n"
        "- Where an explanatory note clarifies how an item is intended to be used "
        "(e.g. a one-off assessment, or that it can be combined with other items), "
        "reflect that and cite the note ID.\n"
        "- Do not declare a single item definitive when others could also apply.\n"
        "- Cite item number, schedule version and source link for each item you mention.\n"
        "- If none clearly fit, say so and point to AskMBS.\n"
        "- Route eligibility and binding interpretation to AskMBS / Services Australia "
        "(13 21 50). General information only, not billing advice."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def call_llm(messages, model=None, temperature=0.0, timeout=60, retries=3):
    """Generate an answer via OpenRouter's OpenAI-compatible chat completions API.

    Uses only the standard library (no extra dependency) so the seam stays light.
    Requires OPENROUTER_API_KEY in the environment. The model defaults to
    DEFAULT_MODEL and can be overridden per-call or via OPENROUTER_MODEL.

    Retries on 429 / 5xx (common on free-tier models when a provider is
    congested), honouring the Retry-After header when present. Raises
    RuntimeError on a missing key or a persistent non-2xx response so callers
    can fall back to assemble_answer().
    """
    import time
    import urllib.request
    import urllib.error

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")

    model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            # Optional but recommended by OpenRouter for attribution/ranking.
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://mbsaiassistant.local"),
            "X-Title": "MBS AI Assistant",
        },
    )

    last_err = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                raise RuntimeError(f"Unexpected OpenRouter response shape: {str(data)[:500]}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            last_err = RuntimeError(f"OpenRouter HTTP {e.code}: {detail}")
            transient = e.code == 429 or 500 <= e.code < 600
            if not transient or attempt == retries:
                raise last_err
            wait = float(e.headers.get("Retry-After") or 0) or (2 ** attempt)
            print(f"  [retry] {e.code} from OpenRouter, waiting {wait:.0f}s "
                  f"(attempt {attempt + 1}/{retries})")
            time.sleep(wait)
        except urllib.error.URLError as e:
            raise RuntimeError(f"OpenRouter request failed: {e.reason}")
    raise last_err


def answer(query, retriever, profession=None, k=5, use_llm=None):
    """End-to-end: retrieve, then generate. Returns (text, retrieved_items).

    use_llm:
      None  -> auto: use OpenRouter iff OPENROUTER_API_KEY is set, else the
               grounded no-LLM template.
      True  -> force the LLM (raises if no key / call fails).
      False -> always use assemble_answer() (no network).

    On any LLM error in auto mode, falls back to the grounded template so the
    tool degrades gracefully rather than erroring out on the practitioner.
    """
    forced = use_llm is True           # caller explicitly demanded the LLM
    nb = not_billable_hit(query)                      # not-billable provider?
    if nb:                                            # explicit answer, skip retrieval
        return explain(nb), []
    retrieved = retriever.retrieve(query, profession, k)
    if use_llm is None:
        use_llm = bool(os.environ.get("OPENROUTER_API_KEY"))
    if not use_llm or not retrieved:
        return assemble_answer(retrieved, profession), retrieved

    messages = build_llm_messages(query, retrieved)
    try:
        return call_llm(messages), retrieved
    except RuntimeError as e:
        if forced:                     # no silent fallback when explicitly forced
            raise
        print(f"  [warn] LLM call failed, using grounded template: {e}")
        return assemble_answer(retrieved, profession), retrieved


# --- evaluation ----------------------------------------------------------------
# Typed eval. Expected item sets are read from the live index at run time, so the bar
# reflects the real schedule, not hardcoded numbers. Case tuple: (type, query, profession, target).
#   recall   -> an IN-PERSON item from target (cat, grp) surfaces in top-k
#   modality -> a TELEHEALTH (video/phone) item from target (cat, grp) surfaces in top-k
#   item     -> the specific target item number surfaces in top-k
#   refuse   -> out of scope: the assistant routes/declines (retrieval clears to nothing)
#   xfail    -> a KNOWN GAP (should route but currently does not); reported, not scored
EVAL_SET = [
    # -- positive recall, one per covered in-person group --
    ("recall", "individual therapy session by a clinical psychologist", "clinical_psychologist", ("8", "M6")),
    ("recall", "focussed psychological strategies by a psychologist", "registered_psychologist", ("8", "M7")),
    ("recall", "focussed psychological strategies by an occupational therapist", "fps_allied_health", ("8", "M7")),
    ("recall", "focussed psychological strategies by a social worker", "fps_allied_health", ("8", "M7")),
    ("recall", "psychiatrist attendance at least 45 minutes", "psychiatrist", ("1", "A8")),
    ("recall", "GP mental health treatment plan 20 to 40 minutes", "general_practitioner", ("1", "A20")),
    ("recall", "GP mental health treatment consultation review", "general_practitioner", ("1", "A20")),
    ("recall", "GP eating disorder treatment and management plan", "general_practitioner", ("1", "A36")),
    ("recall", "dietitian service for an eating disorder", "dietitian", ("8", "M16")),
    ("recall", "eating disorder psychological treatment by a clinical psychologist", "clinical_psychologist", ("8", "M16")),
    # -- modality: telehealth queries should reach the video/phone groups --
    ("modality", "psychiatrist telehealth video attendance", "psychiatrist", ("1", "A40")),
    ("modality", "GP mental health service by telephone", "general_practitioner", ("1", "A40")),
    ("modality", "clinical psychologist video telehealth therapy session", "clinical_psychologist", ("8", "M18")),
    ("modality", "psychologist focussed psychological strategies by phone", "registered_psychologist", ("8", "M18")),
    # -- specific curated item --
    ("item", "initial psychiatric assessment more than 45 minutes new patient", "psychiatrist", "291"),
    # -- refusals the pipeline handles: clinically out-of-scope, no lexical overlap --
    ("refuse", "chest x-ray", "psychiatrist", None),
    ("refuse", "colonoscopy", "general_practitioner", None),
    ("refuse", "hip replacement surgery", "clinical_psychologist", None),
    # -- known gaps (reported, not scored): should route to AskMBS but currently do not --
    ("xfail", "can I claim item 80000 if the patient was not referred", "clinical_psychologist",
     "eligibility question — should route to AskMBS, currently answers with the item"),
    ("xfail", "how many psychology sessions is a patient entitled to in a year", "registered_psychologist",
     "program-rule question — should route, currently retrieves items"),
    ("xfail", "bulk billing rules for a colonoscopy", "general_practitioner",
     "out-of-scope but naturally phrased — generic words (billing/rules) leak past BM25, so it "
     "returns items instead of routing; a relevance floor on generic vocab would close this"),
]

_TAG = {"recall": "RECALL", "modality": "MODAL", "item": "ITEM", "refuse": "REFUSE", "xfail": "XFAIL"}


def evaluate(retriever, k=5):
    inperson, telehealth = {}, {}
    for i, r in enumerate(retriever.recs):
        key = (r["category"], r["group"])
        bucket = inperson if retriever.modality[i] == "in_person" else telehealth
        bucket.setdefault(key, set()).add(r["item_num"])

    def got_for(q, prof):
        return {r["item_num"] for r in retriever.retrieve(q, prof, k)}

    scored = graded = 0
    print(f"\nEVAL (typed, recall@{k}):")
    for typ, q, prof, target in EVAL_SET:
        if typ == "recall":
            got = got_for(q, prof); ok = bool(got & inperson.get(target, set()))
        elif typ == "modality":
            got = got_for(q, prof); ok = bool(got & telehealth.get(target, set()))
        elif typ == "item":
            got = got_for(q, prof); ok = target in got
        elif typ == "refuse":
            got = got_for(q, prof); ok = not got            # nothing clears the floor -> assistant routes
        elif typ == "xfail":
            got = got_for(q, prof); routed = not got
            mark = "ROUTES" if routed else "known-gap"
            print(f"  [{mark:9s}] {_TAG[typ]:6s} {q!r}")
            continue
        scored += 1; graded += ok
        print(f"  [{'PASS' if ok else 'FAIL':9s}] {_TAG[typ]:6s} {q!r}")
    print(f"\n  Score: {graded}/{scored} scored cases passed "
          f"({sum(1 for c in EVAL_SET if c[0]=='xfail')} known-gap cases reported separately).")


def load_dotenv(path=".env"):
    """Minimal .env loader (no dependency). Existing env vars win over the file."""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def main():
    load_dotenv(".env.local")   # local overrides load first (setdefault keeps them)
    load_dotenv(".env")
    if not os.path.exists(CHUNKS):
        print("Run mbs_parser.py first to produce", CHUNKS); return
    r = make_retriever()
    print(f"Indexed {len(r.recs)} items ({SCHEDULE_VERSION}) "
          f"via {type(r).__name__}.")

    llm_on = bool(os.environ.get("OPENROUTER_API_KEY"))
    mode = (f"OpenRouter ({os.environ.get('OPENROUTER_MODEL', DEFAULT_MODEL)})"
            if llm_on else "grounded template (no OPENROUTER_API_KEY set)")
    print(f"Generation mode: {mode}")

    demos = [
        ("individual therapy session for my patient", "clinical_psychologist"),
        ("individual therapy session for my patient", "registered_psychologist"),
        ("initial psychiatric assessment", "psychiatrist"),
    ]
    for q, prof in demos:
        print("\n" + "=" * 78)
        print(f"Q: {q!r}   (as: {prof})")
        text, _ = answer(q, r, profession=prof, k=4)
        print("-" * 78)
        print(text)

    evaluate(r, k=5)


if __name__ == "__main__":
    main()
