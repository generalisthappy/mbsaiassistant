#!/usr/bin/env python3
"""
Compare BM25 vs semantic-embedding retrieval on the real MBS schedule.

Run after mbs_chunks.jsonl and mbs_profession_map.json exist:
    python compare_retrievers.py

Shows:
  1. recall@5 on the grounded EVAL_SET for each backend
  2. a precision spot-check on "initial psychiatric assessment at least 45 minutes"
     — the query where BM25 wrongly surfaced the once-per-lifetime autism item (289)
     instead of a routine initial psychiatric attendance (296 family).
"""

import time
import mbs_rag_prototype as m


def show_eval(r, label):
    print(f"\n=== {label}: recall@5 on EVAL_SET ===")
    m.evaluate(r, k=5)


def precision_check(r, label):
    q = "initial psychiatric assessment at least 45 minutes"
    hits = r.retrieve(q, profession="psychiatrist", k=5)
    print(f"\n--- {label}: top items for {q!r} (as psychiatrist) ---")
    for h in hits:
        print(f"  {h['item_num']:>6}  {' '.join(h['descriptor'].split())[:88]}")


def main():
    print("Building BM25 retriever...")
    t = time.time()
    bm25 = m.BM25Retriever()
    print(f"  ready in {time.time() - t:.1f}s")

    print("Building embedding retriever (first run downloads the model + builds cache)...")
    t = time.time()
    emb = m.EmbeddingRetriever()
    print(f"  ready in {time.time() - t:.1f}s  (cached to {m.EMBED_CACHE} for next time)")

    for r, label in [(bm25, "BM25"), (emb, "EMBEDDING")]:
        show_eval(r, label)
        precision_check(r, label)

    print("\nRead the two precision blocks: the embedding backend should rank a "
          "general initial psychiatric attendance above the niche item 289.")


if __name__ == "__main__":
    main()
