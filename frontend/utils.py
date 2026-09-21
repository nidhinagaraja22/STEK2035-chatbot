"""
Utility helpers for the STEK 2035 Assistant: theming, static metadata
(SDGs, documents, clusters), anonymous feedback storage, exports, and small
render helpers. Keeping these here keeps app.py focused on layout and flow.
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import uuid
from pathlib import Path
from typing import Iterable

import streamlit as st
import streamlit.components.v1 as components

from i18n import t

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
DATA_DIR = BASE_DIR / "data"
STYLES_PATH = BASE_DIR / "styles.css"
FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"
LOGO_PATH = ASSETS_DIR / "logo.png"

LAST_UPDATE = "September 2026"
OFFICIAL_STEK_URL = (
    "https://www.heidelberg.de/HD/Rathaus/stadtentwicklungskonzept+2035.html"
)

# --------------------------------------------------------------------------- #
# Official UN SDG colours + names (1–17). Used for badges/chips.
# --------------------------------------------------------------------------- #
SDG_DATA: dict[int, dict[str, str]] = {
    1:  {"color": "#E5243B", "de": "Keine Armut", "en": "No Poverty"},
    2:  {"color": "#DDA63A", "de": "Kein Hunger", "en": "Zero Hunger"},
    3:  {"color": "#4C9F38", "de": "Gesundheit und Wohlergehen", "en": "Good Health and Well-being"},
    4:  {"color": "#C5192D", "de": "Hochwertige Bildung", "en": "Quality Education"},
    5:  {"color": "#FF3A21", "de": "Geschlechtergleichheit", "en": "Gender Equality"},
    6:  {"color": "#26BDE2", "de": "Sauberes Wasser", "en": "Clean Water and Sanitation"},
    7:  {"color": "#FCC30B", "de": "Bezahlbare und saubere Energie", "en": "Affordable and Clean Energy"},
    8:  {"color": "#A21942", "de": "Menschenwürdige Arbeit und Wachstum", "en": "Decent Work and Economic Growth"},
    9:  {"color": "#FD6925", "de": "Industrie, Innovation und Infrastruktur", "en": "Industry, Innovation and Infrastructure"},
    10: {"color": "#DD1367", "de": "Weniger Ungleichheiten", "en": "Reduced Inequalities"},
    11: {"color": "#FD9D24", "de": "Nachhaltige Städte und Gemeinden", "en": "Sustainable Cities and Communities"},
    12: {"color": "#BF8B2E", "de": "Nachhaltiger Konsum und Produktion", "en": "Responsible Consumption and Production"},
    13: {"color": "#3F7E44", "de": "Maßnahmen zum Klimaschutz", "en": "Climate Action"},
    14: {"color": "#0A97D9", "de": "Leben unter Wasser", "en": "Life Below Water"},
    15: {"color": "#56C02B", "de": "Leben an Land", "en": "Life on Land"},
    16: {"color": "#00689D", "de": "Frieden, Gerechtigkeit und starke Institutionen", "en": "Peace, Justice and Strong Institutions"},
    17: {"color": "#19486A", "de": "Partnerschaften zur Erreichung der Ziele", "en": "Partnerships for the Goals"},
}

# --------------------------------------------------------------------------- #
# STEK thematic clusters (fields of action) — filter options.
# --------------------------------------------------------------------------- #
CLUSTERS: list[dict[str, str]] = [
    {"id": "wohnen",    "de": "Wohnen & Quartiere",           "en": "Housing & Neighbourhoods"},
    {"id": "mobilitaet","de": "Mobilität & Verkehr",          "en": "Mobility & Transport"},
    {"id": "klima",     "de": "Klima, Umwelt & Grünflächen",  "en": "Climate, Environment & Green Space"},
    {"id": "wirtschaft","de": "Wirtschaft & Innovation",      "en": "Economy & Innovation"},
    {"id": "soziales",  "de": "Soziales & Teilhabe",          "en": "Social Affairs & Participation"},
    {"id": "kultur",    "de": "Kultur, Bildung & Freizeit",   "en": "Culture, Education & Leisure"},
]

# --------------------------------------------------------------------------- #
# Indexed source documents (metadata shown in the Documents tab & quick links).
# --------------------------------------------------------------------------- #
DOCUMENTS: list[dict] = [
    {"title": "STEK 2035 – Stadtentwicklungskonzept (Strategie A3)",
     "type_de": "Strategie", "type_en": "Strategy", "date": "2025-07",
     "pages": 60, "url": OFFICIAL_STEK_URL},
    {"title": "STEK Statusbericht",
     "type_de": "Bericht", "type_en": "Report", "date": "2024",
     "pages": 44, "url": OFFICIAL_STEK_URL},
    {"title": "STEK Nachhaltigkeitsbericht 2025",
     "type_de": "Monitoring", "type_en": "Monitoring", "date": "2025",
     "pages": 52, "url": OFFICIAL_STEK_URL},
    {"title": "MRO 2035+ – Modell Räumliche Ordnung (Konzeptbericht)",
     "type_de": "Raumstrategie", "type_en": "Spatial strategy", "date": "2023",
     "pages": 80, "url": OFFICIAL_STEK_URL},
    {"title": "STEK 2035 – Dokumentation Online-Beteiligung 2024",
     "type_de": "Beteiligung", "type_en": "Participation", "date": "2024",
     "pages": 120, "url": OFFICIAL_STEK_URL},
    {"title": "Stadtentwicklungsplan 2015",
     "type_de": "Grundlage", "type_en": "Baseline", "date": "2015",
     "pages": 96, "url": OFFICIAL_STEK_URL},
]


# --------------------------------------------------------------------------- #
# Theming
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    """Inject styles.css once per session."""
    if STYLES_PATH.exists():
        css = STYLES_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def logo_data_uri() -> str | None:
    """Return the logo as a data URI, or None if the asset is missing."""
    if not LOGO_PATH.exists():
        return None
    import base64

    b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _inline_logo_svg() -> str:
    """Fallback vector 'logo' used when assets/logo.png is absent."""
    return (
        '<svg width="52" height="52" viewBox="0 0 52 52" role="img" '
        'aria-label="Heidelberg" xmlns="http://www.w3.org/2000/svg">'
        '<rect width="52" height="52" rx="10" fill="#E2001A"/>'
        '<text x="50%" y="55%" text-anchor="middle" dominant-baseline="middle" '
        'font-family="Source Sans 3, Arial, sans-serif" font-size="26" '
        'font-weight="800" fill="#ffffff">HD</text></svg>'
    )


def render_header(lang: str) -> None:
    """Render the city-style header bar (logo + title + subtitle)."""
    logo = logo_data_uri()
    logo_html = (
        f'<img src="{logo}" alt="{html.escape(t("logo_alt", lang))}" class="hd-logo-img"/>'
        if logo
        else f'<span class="hd-logo-svg">{_inline_logo_svg()}</span>'
    )
    st.markdown(
        f"""
        <header class="app-header" role="banner">
          <div class="app-header-inner">
            <div class="hd-logo">{logo_html}</div>
            <div class="hd-titles">
              <h1 class="hd-title">{html.escape(t("header_title", lang))}</h1>
              <p class="hd-subtitle">{html.escape(t("header_subtitle", lang))}</p>
            </div>
          </div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_disclaimer(lang: str) -> None:
    st.markdown(
        f'<div class="ai-banner" role="note">ℹ️ {html.escape(t("disclaimer_banner", lang))}</div>',
        unsafe_allow_html=True,
    )


def render_footer(lang: str) -> None:
    st.markdown(
        f"""
        <footer class="app-footer" role="contentinfo">
          <nav class="footer-links" aria-label="Footer">
            <a href="https://www.heidelberg.de/impressum" target="_blank" rel="noopener">{html.escape(t("footer_impressum", lang))}</a>
            <a href="https://www.heidelberg.de/datenschutz" target="_blank" rel="noopener">{html.escape(t("footer_datenschutz", lang))}</a>
            <a href="https://www.heidelberg.de/barrierefreiheit" target="_blank" rel="noopener">{html.escape(t("footer_barrierefreiheit", lang))}</a>
            <a href="{OFFICIAL_STEK_URL}" target="_blank" rel="noopener">{html.escape(t("footer_official", lang))}</a>
          </nav>
          <p class="footer-disclaimer">{html.escape(t("footer_ai_disclaimer", lang))}</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# SDG badges
# --------------------------------------------------------------------------- #
def sdg_badges_html(sdgs: Iterable[int], lang: str) -> str:
    """Return HTML chips for the given SDG numbers, coloured officially."""
    chips = []
    for n in sdgs:
        data = SDG_DATA.get(int(n))
        if not data:
            continue
        name = html.escape(data[lang if lang in ("de", "en") else "de"])
        chips.append(
            f'<span class="sdg-chip" style="background:{data["color"]}" '
            f'title="SDG {n}: {name}">SDG {n} · {name}</span>'
        )
    return f'<div class="sdg-row">{"".join(chips)}</div>' if chips else ""


# --------------------------------------------------------------------------- #
# Source rendering
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict], lang: str) -> None:
    """Render the 'Quellen' expander with document / page / snippet / link."""
    if not sources:
        return
    with st.expander(f"📄 {t('sources_label', lang)} ({len(sources)})", expanded=False):
        for s in sources:
            title = html.escape(str(s.get("title", "")))
            page = s.get("page")
            snippet = html.escape(str(s.get("snippet", "")))
            url = s.get("url")
            authority = s.get("authority")
            page_str = f" · {t('source_page', lang)} {page}" if page else ""
            auth_str = (
                f'<span class="src-auth">{html.escape(str(authority))}</span>'
                if authority else ""
            )
            link = (
                f'<a href="{html.escape(url)}" target="_blank" rel="noopener" '
                f'class="src-link">↗ {t("source_open", lang)}</a>'
                if url else ""
            )
            st.markdown(
                f"""
                <div class="src-card">
                  <div class="src-head"><span class="src-title">{title}{page_str}</span>{auth_str}</div>
                  <div class="src-snippet">„{snippet}“</div>
                  {link}
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Copy-to-clipboard button (works inside Streamlit's component iframe)
# --------------------------------------------------------------------------- #
def copy_button(text: str, key: str, label: str, copied_label: str) -> None:
    """Render a small copy button that copies ``text`` to the clipboard."""
    payload = json.dumps(text)
    components.html(
        f"""
        <button id="btn-{key}" class="copy-btn" type="button">📋 {html.escape(label)}</button>
        <script>
          const btn = document.getElementById("btn-{key}");
          btn.addEventListener("click", async () => {{
            const txt = {payload};
            try {{
              await navigator.clipboard.writeText(txt);
            }} catch (e) {{
              const ta = document.createElement("textarea");
              ta.value = txt; document.body.appendChild(ta); ta.select();
              document.execCommand("copy"); document.body.removeChild(ta);
            }}
            btn.textContent = "✓ {html.escape(copied_label)}";
            setTimeout(() => btn.textContent = "📋 {html.escape(label)}", 1800);
          }});
        </script>
        <style>
          .copy-btn {{
            font-family: 'Source Sans 3', system-ui, sans-serif; font-size: 0.85rem;
            border: 1px solid #E0E0E0; background: #fff; color: #1F2933;
            padding: 6px 12px; border-radius: 8px; cursor: pointer;
          }}
          .copy-btn:hover {{ border-color: #E2001A; color: #E2001A; }}
          .copy-btn:focus-visible {{ outline: 3px solid rgba(226,0,26,.4); outline-offset: 2px; }}
        </style>
        """,
        height=44,
    )


# --------------------------------------------------------------------------- #
# Anonymous feedback storage (JSON Lines, no personal data)
# --------------------------------------------------------------------------- #
def save_feedback(rating: str, question: str, answer: str, comment: str,
                  language: str) -> None:
    """Append one anonymous feedback record to data/feedback.jsonl."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "id": uuid.uuid4().hex,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "rating": rating,               # "up" | "down"
        "language": language,
        "question": question,
        "answer": answer,
        "comment": comment.strip(),
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Conversation export (TXT / PDF)
# --------------------------------------------------------------------------- #
def conversation_to_txt(messages: list[dict], lang: str) -> str:
    lines = [t("header_title", lang), t("header_subtitle", lang), "=" * 60, ""]
    role_name = {"user": "Frage" if lang == "de" else "Question",
                 "assistant": "Antwort" if lang == "de" else "Answer"}
    for m in messages:
        lines.append(f"[{role_name.get(m['role'], m['role'])}]")
        lines.append(m["content"].strip())
        if m.get("sources"):
            lines.append(f"-- {t('sources_label', lang)} --")
            for s in m["sources"]:
                pg = f", {t('source_page', lang)} {s.get('page')}" if s.get("page") else ""
                lines.append(f"   • {s.get('title', '')}{pg}")
        lines.append("")
    lines.append("-" * 60)
    lines.append(t("footer_ai_disclaimer", lang))
    return "\n".join(lines)


def conversation_to_pdf(messages: list[dict], lang: str) -> bytes:
    """Render the conversation to a simple, readable PDF (bytes).

    Uses fpdf2 core fonts (Latin-1). German umlauts are covered; any character
    outside Latin-1 is replaced. For full Unicode, register a TTF font.
    """
    from fpdf import FPDF

    def latin(s: str) -> str:
        return s.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_text_color(0xE2, 0x00, 0x1A)
    pdf.set_font("helvetica", "B", 16)
    pdf.multi_cell(0, 8, latin(t("header_title", lang)))
    pdf.set_text_color(0x60, 0x60, 0x60)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, latin(t("header_subtitle", lang)))
    pdf.ln(4)

    role_name = {"user": "Frage" if lang == "de" else "Question",
                 "assistant": "Antwort" if lang == "de" else "Answer"}
    for m in messages:
        pdf.set_text_color(0xE2, 0x00, 0x1A)
        pdf.set_font("helvetica", "B", 12)
        pdf.multi_cell(0, 7, latin(role_name.get(m["role"], m["role"])))
        pdf.set_text_color(0x1F, 0x29, 0x33)
        pdf.set_font("helvetica", "", 11)
        pdf.multi_cell(0, 6, latin(m["content"].strip()))
        if m.get("sources"):
            pdf.set_font("helvetica", "I", 9)
            pdf.set_text_color(0x60, 0x60, 0x60)
            for s in m["sources"]:
                pg = f", {t('source_page', lang)} {s.get('page')}" if s.get("page") else ""
                pdf.multi_cell(0, 5, latin(f"   - {s.get('title', '')}{pg}"))
        pdf.ln(3)

    pdf.set_draw_color(0xE0, 0xE0, 0xE0)
    pdf.set_font("helvetica", "I", 8)
    pdf.set_text_color(0x80, 0x80, 0x80)
    pdf.multi_cell(0, 5, latin(t("footer_ai_disclaimer", lang)))

    out = pdf.output()               # fpdf2 >= 2.7 returns a bytearray
    return bytes(out)
