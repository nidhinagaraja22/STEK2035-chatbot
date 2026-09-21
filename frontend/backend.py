"""
Backend interface for the STEK 2035 Assistant.
================================================

This module is the ONLY contract between the Streamlit UI and the RAG pipeline.
The UI never imports retrieval or LLM code directly — it calls the functions
here. A self-contained **mock** implementation is provided so the front end runs
standalone (``streamlit run app.py``) with no model, index or network.

To plug in the real RAG system, replace the bodies of:
    - health_check()
    - retrieve()
    - answer_query()          (non-streaming)
    - answer_query_stream()   (streaming; set SUPPORTS_STREAMING accordingly)
    - related_sdgs()
    - suggest_followups()
keeping the SAME signatures and return shapes documented below.

Return shape of answer_query():
    {
        "answer":  str,
        "sources": [
            {"title": str, "page": int, "snippet": str, "url": str | None,
             "authority": str | None},
            ...
        ],
        "sdgs":     [int, ...],   # optional — SDG numbers the answer relates to
        "followups": [str, ...],  # optional — up to 3 suggested follow-ups
    }
"""
from __future__ import annotations

import os
import random
import time
from typing import Iterator, TypedDict

# Set STEK_BACKEND_DOWN=1 in the environment to simulate an outage and see the
# UI's error handling. The real backend would replace health_check() entirely.
_SIMULATE_DOWN = os.environ.get("STEK_BACKEND_DOWN", "") == "1"

# The mock streams token-by-token. Set to False to exercise the UI's
# non-streaming fallback path.
SUPPORTS_STREAMING = True

# Seconds between streamed tokens in the mock (keeps the demo lively but calm).
_TOKEN_DELAY = 0.012


class Source(TypedDict):
    title: str
    page: int
    snippet: str
    url: str | None
    authority: str | None


# --------------------------------------------------------------------------- #
# Mock knowledge — a few canonical STEK documents the mock cites.
# --------------------------------------------------------------------------- #
_STEK_URL = "https://www.heidelberg.de/HD/Rathaus/stadtentwicklungskonzept+2035.html"

_MOCK_DOCS = [
    {
        "title": "STEK 2035 – Stadtentwicklungskonzept (Strategie A3)",
        "url": _STEK_URL,
        "authority": "Offizielle STEK-Strategie",
        "snippets": [
            "Das STEK 2035 ist der Wegweiser für eine nachhaltige Entwicklung "
            "Heidelbergs und orientiert sich an den UN-Nachhaltigkeitszielen.",
            "Ziel ist ein sozial gerechtes, klimaneutrales und wirtschaftlich "
            "starkes Heidelberg bis zum Jahr 2035.",
        ],
    },
    {
        "title": "STEK Nachhaltigkeitsbericht 2025",
        "url": _STEK_URL,
        "authority": "Offizieller Bericht",
        "snippets": [
            "Der Nachhaltigkeitsbericht 2025 dokumentiert den Fortschritt anhand "
            "messbarer Indikatoren zu Klima, Mobilität und Wohnen.",
            "Die Treibhausgasemissionen sollen bis 2035 deutlich reduziert werden.",
        ],
    },
    {
        "title": "STEK 2035 – Dokumentation Online-Beteiligung 2024",
        "url": _STEK_URL,
        "authority": "Bürgerbeteiligung",
        "snippets": [
            "Im Beteiligungsprozess wünschten sich viele Bürgerinnen und Bürger "
            "mehr bezahlbaren Wohnraum und sichere Radwege.",
            "Die Rückmeldungen aus der Bürgerschaft flossen in die Zielformulierung "
            "des STEK ein.",
        ],
    },
]

# Keywords -> SDG numbers, so the mock can attach plausible SDG badges.
_SDG_KEYWORDS = {
    "klima": [13, 11], "climate": [13, 11], "co2": [13], "energie": [7], "energy": [7],
    "wohn": [11], "housing": [11], "mobil": [11, 9], "mobility": [11, 9],
    "verkehr": [11], "transport": [11], "grün": [15, 11], "green": [15, 11],
    "wasser": [6], "water": [6], "bildung": [4], "education": [4],
    "gesundheit": [3], "health": [3], "beteilig": [16, 17], "participation": [16, 17],
    "wirtschaft": [8, 9], "economy": [8, 9], "sozial": [10, 11], "social": [10, 11],
    "sdg": [17], "nachhalt": [11, 13], "sustainab": [11, 13],
}

# Questions we treat as out-of-scope in the mock, to demonstrate the
# "not found" fallback (no guessing).
_OUT_OF_SCOPE = ("mannheim", "karlsruhe", "wetter", "weather", "aktien", "stock",
                 "u-bahn", "u bahn")


def _seeded(question: str) -> random.Random:
    """Deterministic RNG so the same question yields the same mock output."""
    return random.Random(hash(question) & 0xFFFFFFFF)


def _is_out_of_scope(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _OUT_OF_SCOPE) or len(q.strip()) < 3


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def health_check() -> bool:
    """Return True if the backend can serve requests.

    Replace with a real ping to your RAG service / vector store / LLM.
    """
    return not _SIMULATE_DOWN


def retrieve(
    question: str,
    *,
    top_k: int = 5,
    topics: dict | None = None,
    language: str = "de",
) -> list[Source]:
    """Return the top-k source passages for a question (no generation).

    In the real system this embeds the query, searches the vector store
    (optionally filtered by ``topics``), and returns chunk metadata.
    """
    if _is_out_of_scope(question):
        return []
    rng = _seeded(question)
    docs = _MOCK_DOCS.copy()
    rng.shuffle(docs)
    sources: list[Source] = []
    for doc in docs[: max(1, min(top_k, len(docs)))]:
        sources.append(
            {
                "title": doc["title"],
                "page": rng.randint(3, 58),
                "snippet": rng.choice(doc["snippets"]),
                "url": doc["url"],
                "authority": doc["authority"],
            }
        )
    return sources


def _compose_answer(
    question: str, sources: list[Source], answer_length: str, language: str
) -> str:
    """Build a plausible, grounded-sounding mock answer from the sources."""
    if not sources:
        # honest abstention — never guess
        from i18n import t

        return t("not_found_fallback", language)

    de = language != "en"
    lead = (
        f"Auf Basis des STEK 2035 lässt sich Ihre Frage „{question.strip()}“ wie "
        "folgt beantworten:"
        if de
        else f"Based on the STEK 2035, your question “{question.strip()}” can be "
        "answered as follows:"
    )
    body_points = []
    for s in sources[:3]:
        body_points.append(
            f"- {s['snippet']} " + (f"(vgl. {s['title']}, S. {s['page']})" if de
                                    else f"(cf. {s['title']}, p. {s['page']})")
        )
    body = "\n".join(body_points)

    if answer_length == "short":
        closing = ""
    else:
        closing = (
            "\n\nInsgesamt verfolgt das STEK 2035 einen integrierten, an den "
            "UN-Nachhaltigkeitszielen ausgerichteten Ansatz, der Klimaschutz, "
            "soziale Gerechtigkeit und wirtschaftliche Entwicklung verbindet."
            if de
            else "\n\nOverall, the STEK 2035 follows an integrated approach aligned "
            "with the UN Sustainable Development Goals, connecting climate action, "
            "social equity and economic development."
        )
    return f"{lead}\n\n{body}{closing}"


def related_sdgs(question: str, sources: list[Source]) -> list[int]:
    """Return SDG numbers (1–17) the answer likely relates to."""
    q = question.lower()
    found: list[int] = []
    for kw, sdgs in _SDG_KEYWORDS.items():
        if kw in q:
            found.extend(sdgs)
    if not found and sources:
        found = [11, 13]  # Sustainable Cities + Climate Action as sensible defaults
    # de-duplicate, keep order, cap at 4 badges
    seen: list[int] = []
    for s in found:
        if s not in seen:
            seen.append(s)
    return seen[:4]


def suggest_followups(question: str, language: str = "de") -> list[str]:
    """Return up to three contextual follow-up questions."""
    if language == "en":
        pool = [
            "Which concrete measures are planned?",
            "How is the progress being measured?",
            "How were citizens involved?",
            "Which SDGs does this relate to?",
            "What is the timeline until 2035?",
        ]
    else:
        pool = [
            "Welche konkreten Maßnahmen sind geplant?",
            "Wie wird der Fortschritt gemessen?",
            "Wie wurden die Bürgerinnen und Bürger einbezogen?",
            "Welche SDGs sind davon betroffen?",
            "Wie ist der Zeitplan bis 2035?",
        ]
    return _seeded(question).sample(pool, 3)


def answer_query(
    question: str,
    history: list[dict],
    language: str,
    *,
    top_k: int = 5,
    answer_length: str = "detailed",
    topics: dict | None = None,
    sources: list[Source] | None = None,
) -> dict:
    """Non-streaming answer. Returns the full response dict (see module docstring).

    ``sources`` may be passed in if already retrieved (to avoid a double lookup);
    otherwise they are fetched here.
    """
    if sources is None:
        sources = retrieve(question, top_k=top_k, topics=topics, language=language)
    answer = _compose_answer(question, sources, answer_length, language)
    return {
        "answer": answer,
        "sources": sources,
        "sdgs": related_sdgs(question, sources),
        "followups": suggest_followups(question, language),
    }


def answer_query_stream(
    question: str,
    history: list[dict],
    language: str,
    *,
    top_k: int = 5,
    answer_length: str = "detailed",
    topics: dict | None = None,
    sources: list[Source] | None = None,
) -> Iterator[str]:
    """Streaming answer generator, yielding text chunks for ``st.write_stream``.

    The mock re-uses answer_query() to compose the full text, then emits it
    word-by-word. A real backend would yield tokens directly from the LLM.
    """
    full = answer_query(
        question, history, language,
        top_k=top_k, answer_length=answer_length, topics=topics, sources=sources,
    )["answer"]
    for token in full.split(" "):
        yield token + " "
        time.sleep(_TOKEN_DELAY)
