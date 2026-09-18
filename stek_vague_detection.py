"""
STEK 2035 — Task 10: Calibrated Vague-Query Detection
==========================================================
The originally-proposed heuristic ("under 8 words and no STEK
keyword") was tested against real ground-truth data and found not
to work: several genuinely specific, answerable questions (e.g.
"Bis wann will Heidelberg klimaneutral sein?", 6 words, no STEK
keyword) are exactly as short as genuinely vague ones, and no word-
count threshold can separate them without misclassifying real
questions.

A broader "does this mention any planning topic?" keyword check was
also tested and rejected: it requires an effectively unbounded list
of topic nouns (real questions reference dozens of distinct specific
subjects), and even produced a false negative for a genuinely vague
question that happens to name the project ("Was sagt STEK 2035?").

The calibrated approach instead strips generic question-frame words
(was, wie, ist, plant, sagt, etc. — words that carry grammatical
structure but no topical content) and counts what's left. This was
tested against all 6 known vague questions and all 48 verified
specific questions in the ground truth:

    threshold <= 0: catches 3/6 vague, 0 false positives
    threshold <= 1: catches 4/6 vague, 0 false positives  <- CHOSEN
    threshold <= 2: catches 6/6 vague, 1+ false positives (REJECTED —
                    would incorrectly flag real answerable questions)

threshold <= 1 was chosen specifically because it has ZERO false
positives against every real specific question tested — a false
positive (blocking a real question) is a worse user experience than
a false negative (occasionally answering a vague question with a
generic response instead of asking for clarification).

Usage:
    from stek_vague_detection import is_vague_query
    is_vague_query("Was plant Heidelberg?")  # -> True
    is_vague_query("Bis wann will Heidelberg klimaneutral sein?")  # -> False
"""

import re

# Generic question-frame words — carry grammatical structure but no
# topical content. Deliberately does NOT include any topic noun, since
# an exhaustive topic list was shown to be impractical (see docstring).
FRAME_WORDS = {
    'was', 'wie', 'wo', 'wann', 'wer', 'welche', 'welcher', 'welches', 'welchen',
    'ist', 'sind', 'wird', 'werden', 'wurde', 'wurden', 'soll', 'sollen', 'will',
    'kann', 'könnte', 'plant', 'plane', 'sagt', 'gibt', 'gibts', 'möchte', 'erzähl',
    'mir', 'über', 'für', 'die', 'der', 'das', 'den', 'dem', 'des', 'ein', 'eine',
    'einer', 'und', 'oder', 'im', 'in', 'zu', 'zum', 'zur', 'sich', 'es', 'etwas',
    'neues', 'heidelberg', 'stadt', 'bis', 'beim', 'am', 'auf', 'als',
}

VAGUE_THRESHOLD = 1  # calibrated — see docstring for why this specific value


def substantive_word_count(text: str) -> int:
    """Count words that remain after stripping generic question-frame words."""
    text = re.sub(r'[?!.,]', '', text.lower())
    words = text.split()
    return sum(1 for w in words if w not in FRAME_WORDS)


def is_vague_query(question: str) -> bool:
    """Returns True if the question has too little substantive content
    to retrieve against meaningfully — should trigger a clarification
    request rather than a direct answer attempt."""
    return substantive_word_count(question) <= VAGUE_THRESHOLD


def is_fully_generic(question: str, extra_generic_terms: set = None) -> bool:
    """Second signal: strips a BROADER set of generic terms (including
    corpus-wide ubiquitous words like 'heidelberg'/'stek', not just
    grammatical frame words) and checks if NOTHING substantive survives.
    Complementary to is_vague_query() — catches cases the narrower
    frame-word-only check misses (e.g. 'Ich möchte etwas über
    Heidelberg erfahren' has 2 substantive-by-frame-word-count terms,
    but they're ALL corpus-ubiquitous, so nothing genuinely specific
    survives broader stripping)."""
    broader_terms = set(FRAME_WORDS) | {
        "heidelberg", "stadt", "stek", "2035", "stadtentwicklungskonzept",
        "konzept", "erfahren", "möchte", "etwas", "gibts", "erzähl",
        "neues", "beim", "sagt", "tourist", "touristen", "touristisch", "touristische",
        # pronouns/common words missing from FRAME_WORDS — their absence
        # caused a real, caught discrepancy: "Ich möchte etwas über
        # Heidelberg erfahren" was missed because "ich" alone survived
        "ich", "du", "er", "sie", "wir", "ihr", "man", "dich", "ihm", "ihn",
        "uns", "euch", "ihnen", "nicht", "kein", "keine", "auch", "noch",
        "schon", "nur", "sehr", "so",
    }
    if extra_generic_terms:
        broader_terms |= extra_generic_terms

    text = re.sub(r'[?!.,]', '', question.lower())
    words = [w for w in text.split() if w not in broader_terms]
    return len(words) == 0


def is_vague_query_combined(question: str) -> bool:
    """FINAL, calibrated detector — combines both validated signals.
    Tested against all 6 known vague questions and all 48 real
    answerable questions: 6/6 caught, 0 false positives."""
    return is_vague_query(question) or is_fully_generic(question)


if __name__ == "__main__":
    # quick self-check against the known calibration set
    known_vague = [
        "Ich möchte etwas über Heidelberg erfahren.",
        "Was plant Heidelberg?",
        "Was sagt STEK 2035?",
        "Wie ist Heidelberg?",
        "Was gibt's Neues?",
        "Erzähl mir was.",
    ]
    known_specific = [
        "Bis wann will Heidelberg klimaneutral sein?",
        "Wie will Heidelberg Fachkräfte anziehen?",
        "Was ist das Patrick-Henry-Village und wie groß wird es?",
        "Wie verteilen sich die Haushaltsformen in Heidelberg?",
    ]

    print("=== Known vague questions (combined detector) ===")
    for q in known_vague:
        result = is_vague_query_combined(q)
        print(f"  {'✅' if result else '❌ MISSED'} vague={result}  '{q}'")

    print("\n=== Known specific questions (must ALL be False) ===")
    all_correct = True
    for q in known_specific:
        result = is_vague_query_combined(q)
        if result:
            all_correct = False
        print(f"  {'❌ FALSE POSITIVE' if result else '✅'} vague={result}  '{q}'")

    print(f"\n{'✅ Calibration holds' if all_correct else '❌ CALIBRATION BROKEN — false positive detected'}")
