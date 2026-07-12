#!/usr/bin/env python3
"""
Not-billable provider registry.

Some provider types come up in mental-health questions but do NOT hold MBS items
in their own right (e.g. credentialed mental health nurses, who are PHN-funded, not
fee-for-service). Rather than let retrieval return nothing (silent exclusion) or the
wrong items, we detect these providers in the query and hand back an explicit,
grounded explanation.

This is deliberately conservative: it only fires when the query both (a) names a
not-billable provider and (b) looks like it's asking about THAT provider's own
items / billing / eligibility — so "a GP reviews a mental health nurse's notes"
does not trip it.
"""

import json
import os
import re

_PATH = os.path.join(os.path.dirname(__file__), "mbs_not_billable.json")

# The query has to look like an item/billing/eligibility question for us to intercept.
_ITEM_QUESTION = re.compile(
    r"\b(item|items|item number|bill|billed|billing|claim|claimed|claimable|"
    r"rebate|medicare|mbs|eligib|provider number)\b",
    re.IGNORECASE,
)


def _load(path=_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["providers"]


_PROVIDERS = _load()


def not_billable_hit(query):
    """Return the provider dict if `query` asks about a not-billable provider's own
    MBS items, else None."""
    q = (query or "").lower()
    if not _ITEM_QUESTION.search(q):
        return None
    for p in _PROVIDERS:
        if any(alias in q for alias in p["aliases"]):
            return p
    return None


def explain(provider):
    """Grounded, practitioner-facing answer for a not-billable provider."""
    return (
        f"{provider['summary']} {provider['funding']} {provider['note']}"
    )


if __name__ == "__main__":
    tests = [
        "what MBS items can a mental health nurse bill?",
        "mental health nurse item numbers",
        "can a credentialed mental health nurse claim a rebate",
        "GP reviews the mental health nurse's care plan",   # should NOT fire
        "psychologist focussed psychological strategies items",  # unrelated
    ]
    for t in tests:
        hit = not_billable_hit(t)
        print(f"{'HIT ' if hit else 'pass'} | {t}")
        if hit:
            print("       ->", explain(hit))
