"""
Internationalisation (i18n) for the STEK 2035 Assistant.

All user-facing strings live here so the UI can be translated by editing a
single file. German ("de") is the default; English ("en") is the fallback.

Usage:
    from i18n import t, LANGUAGES
    t("app_title", lang)                       # -> str
    t("example_questions", lang)               # -> list[str]

Add a new language by adding a top-level key to STRINGS with the same keys.
"""
from __future__ import annotations

from typing import Any

# Display names for the language switcher. Keys are the internal codes.
LANGUAGES: dict[str, str] = {"de": "Deutsch", "en": "English"}
DEFAULT_LANGUAGE = "de"

# --------------------------------------------------------------------------- #
# Translation tables
# --------------------------------------------------------------------------- #
STRINGS: dict[str, dict[str, Any]] = {
    # ===================================================================== #
    # GERMAN (default) — formal register ("Sie")
    # ===================================================================== #
    "de": {
        # --- App / header ---
        "app_title": "STEK 2035 Assistent",
        "header_title": "STEK 2035 – Stadtentwicklungskonzept Heidelberg",
        "header_subtitle": "Fragen Sie den Wegweiser für ein nachhaltiges Heidelberg bis 2035",
        "logo_alt": "Stadt Heidelberg – Logo",

        # --- Disclaimer banner ---
        "disclaimer_banner": (
            "Antworten werden von einer KI auf Basis des STEK 2035 erstellt und "
            "können Fehler enthalten. Bitte prüfen Sie wichtige Angaben in den "
            "Originaldokumenten."
        ),

        # --- Welcome / examples ---
        "welcome_title": "Willkommen beim STEK 2035 Assistenten",
        "welcome_body": (
            "Ich beantworte Ihre Fragen zum Stadtentwicklungskonzept 2035 der "
            "Stadt Heidelberg – der Nachhaltigkeitsstrategie auf Basis der "
            "UN-Nachhaltigkeitsziele (SDGs). Wählen Sie eine Beispielfrage oder "
            "stellen Sie Ihre eigene."
        ),
        "examples_label": "Beispielfragen",
        "example_questions": [
            "Was sind die Hauptziele des STEK 2035?",
            "Wie wird der Klimaschutz im STEK berücksichtigt?",
            "Welche Rolle spielen die SDGs?",
            "Wie lief die Bürgerbeteiligung ab?",
            "Was plant Heidelberg beim bezahlbaren Wohnraum?",
            "Wie soll die Mobilität bis 2035 verbessert werden?",
        ],

        # --- Chat ---
        "chat_input_placeholder": "Stellen Sie Ihre Frage zum STEK 2035 …",
        "thinking": "Ich durchsuche die STEK-Dokumente …",
        "followups_label": "Weiterführende Fragen",
        "sources_label": "Quellen",
        "source_page": "Seite",
        "source_open": "Dokument öffnen",
        "related_sdgs_label": "Verwandte Nachhaltigkeitsziele (SDGs)",

        # --- Answer actions ---
        "copy": "Antwort kopieren",
        "copied": "Kopiert!",
        "regenerate": "Neu generieren",
        "new_conversation": "Neue Unterhaltung",

        # --- Feedback ---
        "feedback_prompt": "War diese Antwort hilfreich?",
        "feedback_up": "Hilfreich",
        "feedback_down": "Nicht hilfreich",
        "feedback_comment_placeholder": "Optionaler Kommentar (anonym) …",
        "feedback_submit": "Feedback senden",
        "feedback_thanks": "Vielen Dank für Ihr Feedback!",

        # --- Export ---
        "export_label": "Unterhaltung exportieren",
        "export_txt": "Als TXT",
        "export_pdf": "Als PDF",
        "export_filename": "STEK2035_Unterhaltung",

        # --- Sidebar ---
        "sidebar_language": "Sprache",
        "sidebar_filters": "Themenfilter",
        "sidebar_clusters": "Themencluster",
        "sidebar_clusters_help": "Antworten auf ausgewählte Handlungsfelder eingrenzen.",
        "sidebar_sdgs": "SDGs (Nachhaltigkeitsziele)",
        "sidebar_sdgs_help": "Auf bestimmte UN-Nachhaltigkeitsziele eingrenzen.",
        "sidebar_advanced": "Erweiterte Einstellungen",
        "sidebar_topk": "Anzahl der Quellen (Top-k)",
        "sidebar_topk_help": "Wie viele Textabschnitte für die Antwort herangezogen werden.",
        "sidebar_length": "Antwortlänge",
        "sidebar_length_short": "Kurz",
        "sidebar_length_detailed": "Ausführlich",
        "sidebar_show_sources": "Quellen anzeigen",
        "sidebar_about": "Über dieses Projekt",
        "sidebar_about_body": (
            "Das **Stadtentwicklungskonzept (STEK) 2035** ist Heidelbergs "
            "Wegweiser für eine nachhaltige Entwicklung bis 2035, ausgerichtet an "
            "den UN-Nachhaltigkeitszielen und im Juli 2025 vom Gemeinderat "
            "beschlossen. Dieser Assistent beantwortet Fragen ausschließlich auf "
            "Basis der offiziellen STEK-Dokumente."
        ),
        "sidebar_docs_used": "Verwendete Dokumente",
        "sidebar_last_update": "Letzte Aktualisierung",
        "sidebar_quick_links": "Schnellzugriff auf Dokumente",

        # --- Tabs ---
        "tab_chat": "Chat",
        "tab_documents": "Dokumente",
        "tab_faq": "Hilfe & FAQ",

        # --- Documents tab ---
        "documents_title": "Indexierte Quelldokumente",
        "documents_intro": (
            "Der Assistent stützt sich ausschließlich auf die folgenden "
            "offiziellen Dokumente der Stadt Heidelberg."
        ),
        "doc_col_title": "Dokument",
        "doc_col_type": "Art",
        "doc_col_date": "Datum",
        "doc_col_pages": "Seiten",
        "doc_col_link": "Link",

        # --- FAQ tab ---
        "faq_title": "Häufig gestellte Fragen",
        "faq_items": [
            {
                "q": "Worauf basieren die Antworten?",
                "a": "Ausschließlich auf den offiziellen STEK-2035-Dokumenten der "
                     "Stadt Heidelberg (u. a. Strategie, Statusbericht, "
                     "Nachhaltigkeitsbericht). Es wird kein allgemeines Weltwissen "
                     "hinzugefügt.",
            },
            {
                "q": "Sind die Antworten rechtlich verbindlich?",
                "a": "Nein. Die Antworten dienen der Information. Verbindlich sind "
                     "allein die Originaldokumente und die Beschlüsse des "
                     "Gemeinderats.",
            },
            {
                "q": "Was passiert, wenn etwas nicht in den Dokumenten steht?",
                "a": "Dann teilt der Assistent ausdrücklich mit, dass die "
                     "vorliegenden STEK-Dokumente dazu keine ausreichenden "
                     "Informationen enthalten – er rät nicht.",
            },
            {
                "q": "Werden meine Daten gespeichert?",
                "a": "Nein. Es werden keine personenbezogenen Daten gespeichert. "
                     "Feedback wird ausschließlich anonym erfasst.",
            },
            {
                "q": "Woher stammen die Seitenangaben in den Quellen?",
                "a": "Jede Quelle verweist auf das Dokument und die Seite, aus der "
                     "der zitierte Abschnitt stammt, damit Sie die Angabe im "
                     "Original prüfen können.",
            },
        ],

        # --- Footer ---
        "footer_impressum": "Impressum",
        "footer_datenschutz": "Datenschutz",
        "footer_barrierefreiheit": "Barrierefreiheit",
        "footer_official": "Offizielle STEK-Seite",
        "footer_ai_disclaimer": (
            "Dies ist ein KI-gestützter Prototyp und kein offizielles Angebot der "
            "Stadt Heidelberg."
        ),

        # --- Errors / fallbacks ---
        "error_backend_down": (
            "Der Dienst ist derzeit nicht erreichbar. Bitte versuchen Sie es in "
            "wenigen Augenblicken erneut."
        ),
        "error_empty": "Zu dieser Anfrage konnten keine Ergebnisse gefunden werden.",
        "not_found_fallback": (
            "Die vorliegenden STEK-Dokumente enthalten nicht genügend "
            "Informationen, um diese Frage zu beantworten."
        ),
    },

    # ===================================================================== #
    # ENGLISH
    # ===================================================================== #
    "en": {
        "app_title": "STEK 2035 Assistant",
        "header_title": "STEK 2035 – Heidelberg Urban Development Concept",
        "header_subtitle": "Ask the guide to a sustainable Heidelberg by 2035",
        "logo_alt": "City of Heidelberg – logo",

        "disclaimer_banner": (
            "Answers are generated by an AI based on the STEK 2035 and may contain "
            "errors. Please verify important information in the original documents."
        ),

        "welcome_title": "Welcome to the STEK 2035 Assistant",
        "welcome_body": (
            "I answer questions about the City of Heidelberg's Urban Development "
            "Concept 2035 — the sustainability strategy based on the UN Sustainable "
            "Development Goals (SDGs). Pick an example question or ask your own."
        ),
        "examples_label": "Example questions",
        "example_questions": [
            "What are the main goals of the STEK 2035?",
            "How is climate protection addressed in the STEK?",
            "What role do the SDGs play?",
            "How was the citizen participation conducted?",
            "What does Heidelberg plan for affordable housing?",
            "How will mobility be improved by 2035?",
        ],

        "chat_input_placeholder": "Ask your question about the STEK 2035 …",
        "thinking": "Searching the STEK documents …",
        "followups_label": "Follow-up questions",
        "sources_label": "Sources",
        "source_page": "Page",
        "source_open": "Open document",
        "related_sdgs_label": "Related Sustainable Development Goals (SDGs)",

        "copy": "Copy answer",
        "copied": "Copied!",
        "regenerate": "Regenerate",
        "new_conversation": "New conversation",

        "feedback_prompt": "Was this answer helpful?",
        "feedback_up": "Helpful",
        "feedback_down": "Not helpful",
        "feedback_comment_placeholder": "Optional comment (anonymous) …",
        "feedback_submit": "Send feedback",
        "feedback_thanks": "Thank you for your feedback!",

        "export_label": "Export conversation",
        "export_txt": "As TXT",
        "export_pdf": "As PDF",
        "export_filename": "STEK2035_conversation",

        "sidebar_language": "Language",
        "sidebar_filters": "Topic filters",
        "sidebar_clusters": "Thematic clusters",
        "sidebar_clusters_help": "Narrow answers to selected fields of action.",
        "sidebar_sdgs": "SDGs (Sustainable Development Goals)",
        "sidebar_sdgs_help": "Narrow to specific UN Sustainable Development Goals.",
        "sidebar_advanced": "Advanced settings",
        "sidebar_topk": "Number of sources (top-k)",
        "sidebar_topk_help": "How many text passages are used to build the answer.",
        "sidebar_length": "Answer length",
        "sidebar_length_short": "Short",
        "sidebar_length_detailed": "Detailed",
        "sidebar_show_sources": "Show sources",
        "sidebar_about": "About this project",
        "sidebar_about_body": (
            "The **Urban Development Concept (STEK) 2035** is Heidelberg's guide to "
            "sustainable development by 2035, aligned with the UN Sustainable "
            "Development Goals and adopted by the city council in July 2025. This "
            "assistant answers questions solely on the basis of the official STEK "
            "documents."
        ),
        "sidebar_docs_used": "Documents used",
        "sidebar_last_update": "Last updated",
        "sidebar_quick_links": "Quick links to documents",

        "tab_chat": "Chat",
        "tab_documents": "Documents",
        "tab_faq": "Help & FAQ",

        "documents_title": "Indexed source documents",
        "documents_intro": (
            "The assistant relies exclusively on the following official documents "
            "of the City of Heidelberg."
        ),
        "doc_col_title": "Document",
        "doc_col_type": "Type",
        "doc_col_date": "Date",
        "doc_col_pages": "Pages",
        "doc_col_link": "Link",

        "faq_title": "Frequently asked questions",
        "faq_items": [
            {
                "q": "What are the answers based on?",
                "a": "Solely on the official STEK 2035 documents of the City of "
                     "Heidelberg (strategy, status report, sustainability report, "
                     "etc.). No general world knowledge is added.",
            },
            {
                "q": "Are the answers legally binding?",
                "a": "No. The answers are for information only. Only the original "
                     "documents and the city council's decisions are binding.",
            },
            {
                "q": "What happens if something is not in the documents?",
                "a": "The assistant explicitly states that the available STEK "
                     "documents do not contain sufficient information — it does not "
                     "guess.",
            },
            {
                "q": "Is my data stored?",
                "a": "No personal data is stored. Feedback is collected anonymously "
                     "only.",
            },
            {
                "q": "Where do the page numbers in the sources come from?",
                "a": "Each source points to the document and page the quoted passage "
                     "comes from, so you can verify it in the original.",
            },
        ],

        "footer_impressum": "Imprint",
        "footer_datenschutz": "Privacy",
        "footer_barrierefreiheit": "Accessibility",
        "footer_official": "Official STEK page",
        "footer_ai_disclaimer": (
            "This is an AI-powered prototype and not an official service of the City "
            "of Heidelberg."
        ),

        "error_backend_down": (
            "The service is currently unavailable. Please try again in a few "
            "moments."
        ),
        "error_empty": "No results could be found for this query.",
        "not_found_fallback": (
            "The available STEK documents do not contain sufficient information to "
            "answer this question."
        ),
    },
}


def t(key: str, lang: str = DEFAULT_LANGUAGE) -> Any:
    """Return the translated value for ``key`` in ``lang``.

    Falls back to German, then to the raw key, so a missing translation never
    crashes the UI.
    """
    lang = lang if lang in STRINGS else DEFAULT_LANGUAGE
    if key in STRINGS[lang]:
        return STRINGS[lang][key]
    if key in STRINGS[DEFAULT_LANGUAGE]:
        return STRINGS[DEFAULT_LANGUAGE][key]
    return key
