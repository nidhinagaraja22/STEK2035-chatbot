"""
calibration_test.py — labeled 20-question test batch for hybrid_gate.py

Covers all five STEK 2035 themes (Wohnen, Mobilitaet, Umwelt, Wirtschaft,
Soziales) plus vague and out-of-domain cases, so REJECT_THRESHOLD is set
from real coverage rather than a handful of permit-related examples.

Usage:
    python calibration_test.py
"""

from hybrid_gate import RelevanceGate, lemmatize_query
from ingestion_pipeline import DOMAIN_STOPWORDS, SYNONYMS
from collections import Counter

# (query, expected_verdict, theme) — expected_verdict is what a human would judge
# correct, used only to score the gate's actual output against.
TEST_CASES = [
    # --- Wohnen ---
    ("Wie viele neue Wohnungen sind bis 2035 in Heidelberg geplant?", "IN_DOMAIN", "Wohnen"),
    ("Welche Massnahmen foerdern bezahlbaren Wohnraum in Heidelberg?", "IN_DOMAIN", "Wohnen"),
    ("Wo entstehen neue Wohngebiete in Heidelberg?", "IN_DOMAIN", "Wohnen"),
    ("Wie unterstuetzt die Stadt sozialen Wohnungsbau?", "IN_DOMAIN", "Wohnen"),

    # --- Mobilitaet ---
    ("Wie plant Heidelberg den Ausbau des oeffentlichen Nahverkehrs?", "IN_DOMAIN", "Mobilitaet"),
    ("Welche Radwege sind im STEK 2035 vorgesehen?", "IN_DOMAIN", "Mobilitaet"),
    ("Wie verbessert die Stadt die Verkehrsanbindung zwischen Heidelberg und Mannheim?", "IN_DOMAIN", "Mobilitaet"),
    ("Gibt es Plaene fuer neue Strassenbahnlinien in Heidelberg?", "IN_DOMAIN", "Mobilitaet"),

    # --- Umwelt ---
    ("Welche Klimaschutzmassnahmen plant Heidelberg bis 2035?", "IN_DOMAIN", "Umwelt"),
    ("Wie will die Stadt die Gruenflaechen erhalten?", "IN_DOMAIN", "Umwelt"),
    ("Welche Massnahmen gibt es zum Hochwasserschutz in Heidelberg?", "IN_DOMAIN", "Umwelt"),

    # --- Wirtschaft ---
    ("Wie foerdert Heidelberg die Ansiedlung neuer Unternehmen?", "IN_DOMAIN", "Wirtschaft"),
    ("Welche Rolle spielt Wissenschaft und Forschung fuer die Stadtentwicklung?", "IN_DOMAIN", "Wirtschaft"),
    ("Wie unterstuetzt die Stadt die Gewinnung von Fachkraeften?", "IN_DOMAIN", "Wirtschaft"),

    # --- Soziales / Kultur ---
    ("Welche kulturellen Angebote plant die Stadt fuer die Zukunft?", "IN_DOMAIN", "Soziales"),
    ("Wie foerdert Heidelberg gesellschaftliche Vielfalt?", "IN_DOMAIN", "Soziales"),
    ("Welche sozialen Projekte gibt es fuer Senioren in Heidelberg?", "IN_DOMAIN", "Soziales"),

    # --- Vague ---
    ("Erzaehl mir etwas ueber Heidelberg", "VAGUE", "vague"),

    # --- Out-of-domain ---
    ("Wie bereite ich einen klassischen Kartoffelsalat zu?", "OUT_OF_DOMAIN", "off-topic"),
    ("Welche Genehmigungen brauche ich, um in Muenchen zu wohnen?", "OUT_OF_DOMAIN", "off-topic"),
]


def suggest_synonym_candidates(gate: RelevanceGate, query: str, top_n_chunks: int = 5,
                                 min_frequency: int = 2) -> list[tuple[str, int]]:
    """
    For a query that SHOULD be IN_DOMAIN but scored low, compares the query's
    own lemma tokens against the tokens actually present in its top-N
    retrieved chunks. Words that appear in several of those chunks but NOT
    in the query itself are candidate synonyms — the corpus's own wording
    for the same concept, which BM25 can't currently connect to the query.

    This automates what we did manually for "Grünflächen" -> "Stadtgrün" /
    "Begrünung": those exact terms would have shown up here as high-frequency
    non-query words across the top chunks.

    Returns [(candidate_word, chunk_count)], sorted by frequency, excluding:
    - words already in the query itself
    - words already mapped in SYNONYMS (no need to suggest twice)
    - DOMAIN_STOPWORDS (boilerplate/administrative noise)
    - single-character tokens or pure numbers (OCR/formatting artifacts)
    """
    query_tokens = set(lemmatize_query(query))
    result = gate.check(query, top_n_chunks=top_n_chunks)

    candidate_counter = Counter()
    for chunk in result["top_chunks"]:
        chunk_tokens = set(chunk["tokens"])
        new_words = chunk_tokens - query_tokens
        for word in new_words:
            if word in SYNONYMS or word in DOMAIN_STOPWORDS:
                continue
            if len(word) <= 2 or word.replace(".", "").isdigit():
                continue
            candidate_counter[word] += 1

    candidates = [(w, c) for w, c in candidate_counter.items() if c >= min_frequency]
    candidates.sort(key=lambda x: -x[1])
    return candidates


def print_synonym_suggestions(gate: RelevanceGate, failures: list[dict]):
    """Prints ready-to-review SYNONYMS entries for every misclassified query
    that was expected to be IN_DOMAIN — the cases where the fix is corpus
    vocabulary, not scoring logic."""
    in_domain_failures = [r for r in failures if r["expected"] == "IN_DOMAIN"]
    if not in_domain_failures:
        return

    print(f"\n{'='*100}")
    print("SYNONYM CANDIDATES — words the corpus uses that the query doesn't")
    print(f"{'='*100}")
    print("Review each before adding — these are frequency-based suggestions,")
    print("not verified synonyms. Pick the right canonical target yourself.\n")

    for r in in_domain_failures:
        candidates = suggest_synonym_candidates(gate, r["query"])
        if not candidates:
            print(f"Query: {r['query']}")
            print("  No repeated non-query words found across top chunks — "
                  "likely a genuine thin-coverage gap, not a vocabulary mismatch.\n")
            continue

        query_tokens = sorted(lemmatize_query(r["query"]))
        print(f"Query: {r['query']}")
        print(f"  Query lemmas: {query_tokens}")
        print(f"  Candidate corpus words (word: chunks containing it):")
        for word, count in candidates[:8]:
            print(f"    {word:20} ({count}/5 top chunks)")

        # Best-effort starter suggestion: map the candidates to the longest
        # query lemma (usually the core compound noun) — EDIT before using.
        main_term = max(query_tokens, key=len) if query_tokens else "???"
        print(f"  Starter SYNONYMS lines to review (target guessed as '{main_term}'):")
        for word, _ in candidates[:5]:
            print(f'    "{word}": "{main_term}",')
        print()


def run_calibration(show_diagnostics_for_failures: bool = True,
                     show_synonym_suggestions: bool = True):
    gate = RelevanceGate()

    results = []
    for query, expected, theme in TEST_CASES:
        result = gate.check(query)
        actual = result["verdict"]
        correct = (actual == expected)
        results.append({
            "query": query,
            "theme": theme,
            "expected": expected,
            "actual": actual,
            "score": result["top_score"],
            "correct": correct,
        })

    print(f"{'Theme':12} {'Score':7} {'Expected':14} {'Actual':14} {'OK?':4}  Query")
    print("-" * 100)
    for r in results:
        mark = "OK" if r["correct"] else "XX"
        print(f"{r['theme']:12} {r['score']:.4f} {r['expected']:14} {r['actual']:14} {mark:4}  {r['query'][:50]}")

    correct_count = sum(r["correct"] for r in results)
    print(f"\nAccuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.0f}%)")

    # Show the score gap between correctly-classified in-domain vs out-of-domain,
    # so you can see whether REJECT_THRESHOLD sits in a safe gap or a tight one.
    in_domain_scores = [r["score"] for r in results if r["expected"] == "IN_DOMAIN"]
    off_domain_scores = [r["score"] for r in results if r["expected"] == "OUT_OF_DOMAIN"]
    if in_domain_scores and off_domain_scores:
        print(f"\nLowest IN_DOMAIN score:  {min(in_domain_scores):.4f}")
        print(f"Highest OUT_OF_DOMAIN score: {max(off_domain_scores):.4f}")
        gap = min(in_domain_scores) - max(off_domain_scores)
        print(f"Gap: {gap:.4f}  ({'safe margin' if gap > 0.03 else 'TIGHT — consider more test cases'})")

    # Diagnostic breakdown for every misclassified query — shows the top-5
    # chunks and their individual bm25/cos contributions, so you can see
    # WHY each failure happened (corpus gap vs. terminology mismatch vs.
    # genuine overlap) instead of only seeing the final wrong verdict.
    if show_diagnostics_for_failures:
        failures = [r for r in results if not r["correct"]]
        if failures:
            print(f"\n{'='*100}")
            print(f"DIAGNOSTICS FOR {len(failures)} MISCLASSIFIED QUERIES")
            print(f"{'='*100}")
            for r in failures:
                gate.diagnose(r["query"])

            if show_synonym_suggestions:
                print_synonym_suggestions(gate, failures)

    return results


if __name__ == "__main__":
    run_calibration()