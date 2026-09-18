"""
STEK 2035 — New Strategy: Authority-Primary + Source/Topic Capping
========================================================================
Combines two independently-validated mechanisms into one new,
10th strategy, not present in the original Task 3 comparison:

  1. Authority-primary sort (from this session's cascading design):
     sort ALL candidates by (authority_level ASC, similarity DESC)
     — official content always ranks above citizen content,
     regardless of similarity score; similarity only breaks ties
     WITHIN the same authority level.

  2. Source+topic dual capping (from Task 3's combined_source_topic,
     confirmed as the best all-round performer once authority is
     handled separately): the same validated 3-stage greedy fill
     — strict (both caps), relaxed (source cap only), last-resort
     (no caps) — but now walking down the AUTHORITY-PRIMARY sorted
     list instead of a pure-similarity sorted list.

This directly targets the two known failure patterns from this
session's full 45-question evaluation:
  - Q034/Q037 (within-tier source flooding) — fixed by the source cap
  - Generic-overview-content questions — NOT fixed by this (that
    requires reranking, a separate, larger effort — documented as
    an accepted limitation)

Usage:
    from stek_strategy_authority_source_topic import authority_source_topic_strategy
    top5 = authority_source_topic_strategy(candidates)

Each candidate dict must have: similarity, authority_level, source
(document_id), topic (LDA topic label).
"""

from collections import defaultdict

TOP_K = 5
MAX_PER_SOURCE = 2
MAX_PER_TOPIC = 2


def authority_source_topic_strategy(candidates: list, max_per_source: int = MAX_PER_SOURCE,
                                       max_per_topic: int = MAX_PER_TOPIC, k: int = TOP_K) -> list:
    """
    Sort by (authority_level ASC, similarity DESC) — authority is the
    PRIMARY key, so a lower-authority chunk NEVER outranks a
    higher-authority chunk regardless of similarity score. Similarity
    only decides ordering within the same authority level.

    Then apply the same validated 3-stage greedy fill as
    combined_source_topic, walking down this authority-primary list.
    """
    ranked = sorted(candidates, key=lambda c: (c["authority_level"], -c["similarity"]))

    selected = []
    selected_ids = set()
    source_count = defaultdict(int)
    topic_count = defaultdict(int)

    # stage 1 — strict: both caps enforced
    for c in ranked:
        if len(selected) == k:
            break
        if (source_count[c["source"]] < max_per_source and
                topic_count[c["topic"]] < max_per_topic):
            selected.append(c)
            selected_ids.add(id(c))
            source_count[c["source"]] += 1
            topic_count[c["topic"]] += 1

    # stage 2 — relax topic cap, keep source cap
    if len(selected) < k:
        for c in ranked:
            if len(selected) == k:
                break
            if id(c) in selected_ids:
                continue
            if source_count[c["source"]] < max_per_source:
                selected.append(c)
                selected_ids.add(id(c))
                source_count[c["source"]] += 1

    # stage 3 — last resort, ignore both caps, fill by (authority, similarity) order
    if len(selected) < k:
        for c in ranked:
            if len(selected) == k:
                break
            if id(c) in selected_ids:
                continue
            selected.append(c)
            selected_ids.add(id(c))

    return selected


if __name__ == "__main__":
    # controlled test — mirrors the exact scenario used to explain this
    # design earlier: does authority correctly override raw similarity,
    # and does the source cap correctly prevent flooding?
    test_candidates = [
        {"id": "A", "similarity": 0.72, "authority_level": 1, "source": "stek_a3", "topic": "housing"},
        {"id": "B", "similarity": 0.81, "authority_level": 2, "source": "mro", "topic": "climate"},
        {"id": "C", "similarity": 0.79, "authority_level": 2, "source": "mro", "topic": "climate"},
        {"id": "D", "similarity": 0.77, "authority_level": 2, "source": "mro", "topic": "mobility"},
        {"id": "E", "similarity": 0.85, "authority_level": 4, "source": "arbeitstreffen", "topic": "culture"},
        {"id": "F", "similarity": 0.90, "authority_level": 5, "source": "online_beteiligung", "topic": "housing"},
        {"id": "G", "similarity": 0.83, "authority_level": 5, "source": "zukunftsreise", "topic": "culture"},
    ]

    result = authority_source_topic_strategy(test_candidates)
    print("Selected (in order):")
    for c in result:
        print(f"  {c['id']}: sim={c['similarity']} L{c['authority_level']} source={c['source']} topic={c['topic']}")

    print("\n--- Verification ---")
    ids = [c["id"] for c in result]
    print(f"Result IDs: {ids}")
    print(f"Expected: A, B, C, E, F")
    print(f"(D is correctly EXCLUDED: B, C, D are all source='mro' — the")
    print(f"source cap of 2 kicks in at D, correctly preventing mro from")
    print(f"taking 3 of 5 slots even though it's official-tier content.")
    print(f"F then correctly fills the 5th slot since its source and topic")
    print(f"are both still under cap, even though it's L5.)")
    assert ids == ["A", "B", "C", "E", "F"], f"MISMATCH — got {ids}"
    print("✅ Matches expected behavior exactly")

    print(f"\nNote: G (zukunftsreise, L5) was excluded despite having a higher")
    print(f"similarity (0.83) than F is needed to beat — this is because F")
    print(f"was reached FIRST in the authority-then-similarity sort order")
    print(f"(F: 0.90 vs G: 0.83, both L5), and filled the last slot before")
    print(f"G was ever checked. This confirms similarity still matters AS")
    print(f"A TIE-BREAKER within the same authority level, exactly as designed.")
