# STEK 2035 Assistant — Streamlit Front End

A polished, accessible chat UI for the City of Heidelberg's **Stadtentwicklungskonzept
2035** (STEK 2035) — the sustainability roadmap to 2035 based on the UN SDGs,
adopted by the Gemeinderat in July 2025.

The UI is **decoupled from the RAG pipeline**: it talks to the backend only
through `backend.py`. A mock implementation ships with it, so the app runs
standalone with no model, index, or network.

![Heidelberg red civic theme](assets/logo.png)

---

## Features

- **Chat** with `st.chat_message` / `st.chat_input`, streaming answers
  (`st.write_stream`) with a non-streaming fallback, and a "thinking" spinner.
- **Welcome screen** with 4–6 clickable example questions (localised).
- **Source citations** under each answer (`Quellen` expander): document, page,
  snippet, authority tag, and a link to the source.
- **Per-answer feedback** (👍 / 👎 + optional comment) saved anonymously to
  `data/feedback.jsonl`.
- **Copy answer**, **Regenerate**, and **New conversation**.
- **Export** the conversation as **TXT** or **PDF**.
- **Suggested follow-up questions** (3 chips) after each answer.
- **Sidebar**: language switcher (DE/EN), topic filters (STEK clusters + SDGs),
  advanced settings (top-k, answer length, show/hide sources), an "About"
  section, and quick links to the documents.
- **Tabs**: `Chat` · `Dokumente` (indexed documents + metadata) · `Hilfe & FAQ`.
- **SDG badges** in official SDG colours when an answer relates to specific goals.
- **Accessibility**: WCAG-AA contrast, visible focus states, alt texts, reduced-
  motion support, responsive down to ~375 px, formal German ("Sie").
- **Privacy-friendly**: no personal data stored; feedback is anonymous.

---

## Project structure

```
frontend/
├── app.py                 # main Streamlit app (layout, state, flow)
├── backend.py             # RAG interface + mock (the ONLY integration point)
├── i18n.py                # all UI strings (de / en)
├── utils.py               # theming, SDG/doc metadata, feedback, exports, helpers
├── styles.css             # fine-grained CSS (injected at runtime)
├── .streamlit/config.toml # Streamlit theme (colours, font)
├── assets/logo.png        # placeholder logo (replace with the official one)
├── tools/make_logo.py     # regenerates the placeholder logo (needs Pillow)
├── data/                  # anonymous feedback is written here at runtime
├── requirements.txt
└── README.md
```

---

## Setup

```bash
cd frontend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

> **PDF export** uses `fpdf2` (in `requirements.txt`). If it isn't installed,
> the PDF button is hidden gracefully and TXT export still works.

---

## Plugging in the real RAG backend

Open `backend.py` and replace the bodies of these functions, **keeping the same
signatures and return shapes**:

| Function | Replace with |
|---|---|
| `health_check() -> bool` | a real ping to your RAG service / vector store / LLM |
| `retrieve(question, *, top_k, topics, language) -> list[Source]` | your vector search (honour `topics` as a metadata filter) |
| `answer_query(...) -> dict` | full (non-streaming) generation |
| `answer_query_stream(...) -> Iterator[str]` | token streaming from your LLM |
| `related_sdgs(question, sources) -> list[int]` | SDG tagging (or return `[]`) |
| `suggest_followups(question, language) -> list[str]` | follow-up generation (or return `[]`) |

`answer_query()` must return:

```python
{
  "answer": str,
  "sources": [
    {"title": str, "page": int, "snippet": str,
     "url": str | None, "authority": str | None},
    ...
  ],
  "sdgs": [int, ...],       # optional
  "followups": [str, ...],  # optional
}
```

Set `SUPPORTS_STREAMING = True/False` at the top of `backend.py` to control
whether the UI streams. If streaming raises, the UI automatically falls back to
`answer_query()`.

**Honest abstention:** if `retrieve()` returns an empty list, the UI shows the
"not enough information in the STEK documents" fallback instead of guessing —
keep that behaviour in the real backend for a trustworthy public-sector system.

### Testing error handling
Run with `STEK_BACKEND_DOWN=1` to simulate an outage and see the friendly error:

```bash
# Windows PowerShell
$env:STEK_BACKEND_DOWN=1; streamlit run app.py
# macOS/Linux
STEK_BACKEND_DOWN=1 streamlit run app.py
```

---

## Changing the theme colours

Colours are defined in **two** places that must stay in sync:

1. `.streamlit/config.toml` — `primaryColor`, `backgroundColor`,
   `secondaryBackgroundColor`, `textColor`.
2. `styles.css` — the CSS variables in the `:root { … }` block
   (`--hd-red`, `--hd-text`, `--hd-bg`, `--hd-bg-2`, `--hd-border`, …).

Change both, then reload. To swap the logo, replace `assets/logo.png` (or run
`python tools/make_logo.py` to regenerate the placeholder). The header falls
back to an inline SVG badge if no logo file is present.

## Adding a language

Add a new top-level key to `STRINGS` in `i18n.py` with the same keys as `"de"`,
then add it to `LANGUAGES`. Everything else (sidebar, tabs, examples, FAQ)
updates automatically.

---

## Privacy

No personal data is collected or stored. Feedback is anonymous
(`data/feedback.jsonl`: rating, language, question, answer, optional comment,
timestamp) and is git-ignored. Usage statistics are disabled in
`.streamlit/config.toml`.
