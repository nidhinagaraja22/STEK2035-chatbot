"""
STEK 2035 Assistant — Streamlit front end
==========================================

A polished, accessible chat UI for the City of Heidelberg's Urban Development
Concept (STEK) 2035. All retrieval/LLM logic lives behind ``backend.py``; this
file is layout, state and flow only.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import uuid

import streamlit as st

import backend
import utils
from i18n import DEFAULT_LANGUAGE, LANGUAGES, t

ASSISTANT_AVATAR = "🏛️"
USER_AVATAR = "👤"


# --------------------------------------------------------------------------- #
# Page config + one-time setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="STEK 2035 Assistent",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_state() -> None:
    """Initialise all session_state keys exactly once."""
    ss = st.session_state
    ss.setdefault("lang", DEFAULT_LANGUAGE)
    ss.setdefault("messages", [])          # list[dict]: role, content, sources, sdgs, followups, id, question
    ss.setdefault("pending", None)         # a queued question (from example/follow-up chips)
    ss.setdefault("regen", None)           # a question to regenerate
    ss.setdefault("feedback_given", {})    # msg_id -> "up"|"down"
    ss.setdefault("show_comment", {})      # msg_id -> bool


init_state()
utils.inject_css()
lang = st.session_state.lang


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    """Render the sidebar and return the current retrieval settings."""
    with st.sidebar:
        st.markdown('<div class="sidebar-brand">STEK 2035</div>', unsafe_allow_html=True)

        # --- Language switcher (changing it reruns the app) ---
        st.selectbox(
            t("sidebar_language", lang),
            options=list(LANGUAGES.keys()),
            format_func=lambda code: LANGUAGES[code],
            key="lang",
        )
        cur = st.session_state.lang

        st.divider()

        # --- Topic filters ---
        st.markdown(f"**{t('sidebar_filters', cur)}**")
        cluster_options = [c["id"] for c in utils.CLUSTERS]
        cluster_labels = {c["id"]: c[cur] for c in utils.CLUSTERS}
        st.multiselect(
            t("sidebar_clusters", cur),
            options=cluster_options,
            format_func=lambda cid: cluster_labels[cid],
            help=t("sidebar_clusters_help", cur),
            key="clusters",
        )
        sdg_options = list(utils.SDG_DATA.keys())
        st.multiselect(
            t("sidebar_sdgs", cur),
            options=sdg_options,
            format_func=lambda n: f"SDG {n} · {utils.SDG_DATA[n][cur]}",
            help=t("sidebar_sdgs_help", cur),
            key="sdgs",
        )

        # --- Advanced settings ---
        with st.expander(t("sidebar_advanced", cur), expanded=False):
            st.slider(t("sidebar_topk", cur), 1, 10, 5, key="topk",
                      help=t("sidebar_topk_help", cur))
            st.radio(
                t("sidebar_length", cur),
                options=["short", "detailed"],
                index=1,
                format_func=lambda v: t(f"sidebar_length_{v}", cur),
                key="length",
                horizontal=True,
            )
            st.toggle(t("sidebar_show_sources", cur), value=True, key="show_sources")

        st.divider()

        # --- About ---
        st.markdown(f"**{t('sidebar_about', cur)}**")
        st.markdown(t("sidebar_about_body", cur))
        st.caption(f"**{t('sidebar_docs_used', cur)}:** "
                   "STEK 2035, Statusbericht, Nachhaltigkeitsbericht 2025")
        st.caption(f"**{t('sidebar_last_update', cur)}:** {utils.LAST_UPDATE}")

        # --- Quick links ---
        with st.expander(t("sidebar_quick_links", cur), expanded=False):
            for d in utils.DOCUMENTS:
                st.markdown(f"- [{d['title']}]({d['url']})")

        st.divider()

        # --- Export + reset ---
        if st.session_state.messages:
            st.markdown(f"**{t('export_label', cur)}**")
            col1, col2 = st.columns(2)
            txt = utils.conversation_to_txt(st.session_state.messages, cur)
            col1.download_button(
                t("export_txt", cur), data=txt.encode("utf-8"),
                file_name=f"{t('export_filename', cur)}.txt",
                mime="text/plain", use_container_width=True,
            )
            try:
                pdf_bytes = utils.conversation_to_pdf(st.session_state.messages, cur)
                col2.download_button(
                    t("export_pdf", cur), data=pdf_bytes,
                    file_name=f"{t('export_filename', cur)}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            except Exception:
                col2.caption("PDF: install fpdf2")

        if st.button(f"🗑️ {t('new_conversation', cur)}", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback_given = {}
            st.session_state.show_comment = {}
            st.rerun()

    return {
        "top_k": st.session_state.get("topk", 5),
        "answer_length": st.session_state.get("length", "detailed"),
        "topics": {
            "clusters": st.session_state.get("clusters", []),
            "sdgs": st.session_state.get("sdgs", []),
        },
        "show_sources": st.session_state.get("show_sources", True),
    }


# --------------------------------------------------------------------------- #
# Chat rendering
# --------------------------------------------------------------------------- #
def render_welcome() -> None:
    st.markdown(
        f"""
        <div class="welcome-card">
          <h2>{t('welcome_title', lang)}</h2>
          <p>{t('welcome_body', lang)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"**{t('examples_label', lang)}**")
    examples = t("example_questions", lang)
    cols = st.columns(2)
    for i, q in enumerate(examples):
        if cols[i % 2].button(q, key=f"ex_{i}", type="secondary",
                              use_container_width=True):
            st.session_state.pending = q
            st.rerun()


def render_followups(followups: list[str], msg_id: str) -> None:
    if not followups:
        return
    st.markdown(f'<div class="followups-label">💡 {t("followups_label", lang)}</div>',
                unsafe_allow_html=True)
    cols = st.columns(len(followups))
    for i, q in enumerate(followups):
        if cols[i].button(q, key=f"fu_{msg_id}_{i}", type="secondary",
                          use_container_width=True):
            st.session_state.pending = q
            st.rerun()


def render_feedback(msg: dict) -> None:
    mid = msg["id"]
    if st.session_state.feedback_given.get(mid):
        st.caption(f"✓ {t('feedback_thanks', lang)}")
        return
    st.caption(t("feedback_prompt", lang))
    c1, c2, _ = st.columns([1, 1, 8])
    if c1.button(f"👍", key=f"up_{mid}", help=t("feedback_up", lang)):
        utils.save_feedback("up", msg.get("question", ""), msg["content"], "", lang)
        st.session_state.feedback_given[mid] = "up"
        st.rerun()
    if c2.button(f"👎", key=f"down_{mid}", help=t("feedback_down", lang)):
        st.session_state.show_comment[mid] = True
    if st.session_state.show_comment.get(mid):
        comment = st.text_area(
            t("feedback_down", lang),
            key=f"cmt_{mid}", label_visibility="collapsed",
            placeholder=t("feedback_comment_placeholder", lang),
        )
        if st.button(t("feedback_submit", lang), key=f"sub_{mid}", type="primary"):
            utils.save_feedback("down", msg.get("question", ""), msg["content"],
                                comment or "", lang)
            st.session_state.feedback_given[mid] = "down"
            st.session_state.show_comment[mid] = False
            st.rerun()


def render_message(msg: dict, is_last: bool, show_sources: bool) -> None:
    """Render one stored message (with actions for assistant messages)."""
    if msg["role"] == "user":
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(msg["content"])
        return

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        st.markdown(msg["content"])

        # SDG badges
        if msg.get("sdgs"):
            st.markdown(f'<div class="sdg-label">{t("related_sdgs_label", lang)}</div>',
                        unsafe_allow_html=True)
            st.markdown(utils.sdg_badges_html(msg["sdgs"], lang), unsafe_allow_html=True)

        # Sources
        if show_sources and msg.get("sources"):
            utils.render_sources(msg["sources"], lang)

        # Action row: copy (last only) + regenerate (last only) + feedback (all)
        if is_last:
            a1, a2, _ = st.columns([2, 2, 6])
            with a1:
                utils.copy_button(msg["content"], key=msg["id"],
                                  label=t("copy", lang), copied_label=t("copied", lang))
            with a2:
                if st.button(f"🔄 {t('regenerate', lang)}", key=f"regen_{msg['id']}"):
                    st.session_state.regen = msg.get("question", "")
                    st.session_state.messages.pop()  # drop this assistant answer
                    st.rerun()

        render_feedback(msg)

        # Follow-up suggestions only under the most recent answer
        if is_last and msg.get("followups"):
            render_followups(msg["followups"], msg["id"])


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate_answer(question: str, settings: dict) -> None:
    """Retrieve + generate for ``question`` and append the result to history.

    Renders the assistant reply live (streaming). A final st.rerun() by the
    caller re-renders everything uniformly from history (with action buttons).
    """
    history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        answer, sources, sdgs, followups = "", [], [], []
        try:
            if not backend.health_check():
                st.error(t("error_backend_down", lang))
                answer = t("error_backend_down", lang)
            else:
                with st.spinner(t("thinking", lang)):
                    sources = backend.retrieve(
                        question, top_k=settings["top_k"],
                        topics=settings["topics"], language=lang,
                    )
                if not sources:
                    # honest fallback — no guessing
                    answer = t("not_found_fallback", lang)
                    st.markdown(answer)
                elif backend.SUPPORTS_STREAMING:
                    try:
                        answer = st.write_stream(
                            backend.answer_query_stream(
                                question, history, lang,
                                top_k=settings["top_k"],
                                answer_length=settings["answer_length"],
                                topics=settings["topics"], sources=sources,
                            )
                        )
                    except Exception:                      # streaming failed -> fallback
                        resp = backend.answer_query(
                            question, history, lang,
                            top_k=settings["top_k"],
                            answer_length=settings["answer_length"],
                            topics=settings["topics"], sources=sources,
                        )
                        answer = resp["answer"]
                        st.markdown(answer)
                else:
                    resp = backend.answer_query(
                        question, history, lang,
                        top_k=settings["top_k"],
                        answer_length=settings["answer_length"],
                        topics=settings["topics"], sources=sources,
                    )
                    answer = resp["answer"]
                    st.markdown(answer)

                if sources:
                    sdgs = backend.related_sdgs(question, sources)
                    followups = backend.suggest_followups(question, lang)
        except Exception:
            st.error(t("error_backend_down", lang))
            answer = t("error_backend_down", lang)

    st.session_state.messages.append({
        "id": uuid.uuid4().hex,
        "role": "assistant",
        "content": answer,
        "question": question,
        "sources": sources,
        "sdgs": sdgs,
        "followups": followups,
    })


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def chat_tab(settings: dict) -> None:
    messages = st.session_state.messages

    if not messages:
        render_welcome()

    for i, msg in enumerate(messages):
        render_message(msg, is_last=(i == len(messages) - 1),
                       show_sources=settings["show_sources"])

    # Handle a queued regeneration (no new user bubble).
    regen_q = st.session_state.pop("regen", None)
    if regen_q:
        generate_answer(regen_q, settings)
        st.rerun()

    # Chat input + queued example/follow-up questions.
    typed = st.chat_input(t("chat_input_placeholder", lang))
    prompt = typed or st.session_state.pop("pending", None)
    if prompt:
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)
        st.session_state.messages.append({
            "id": uuid.uuid4().hex, "role": "user", "content": prompt,
        })
        generate_answer(prompt, settings)
        st.rerun()


def documents_tab() -> None:
    st.subheader(t("documents_title", lang))
    st.write(t("documents_intro", lang))
    type_key = "type_de" if lang == "de" else "type_en"
    rows = "".join(
        f"<tr><td>{d['title']}</td><td>{d[type_key]}</td><td>{d['date']}</td>"
        f"<td>{d['pages']}</td>"
        f"<td><a href='{d['url']}' target='_blank' rel='noopener'>↗</a></td></tr>"
        for d in utils.DOCUMENTS
    )
    st.markdown(
        f"""
        <table class="doc-table">
          <thead><tr>
            <th>{t('doc_col_title', lang)}</th><th>{t('doc_col_type', lang)}</th>
            <th>{t('doc_col_date', lang)}</th><th>{t('doc_col_pages', lang)}</th>
            <th>{t('doc_col_link', lang)}</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )


def faq_tab() -> None:
    st.subheader(t("faq_title", lang))
    for item in t("faq_items", lang):
        with st.expander(item["q"]):
            st.write(item["a"])


# --------------------------------------------------------------------------- #
# Main layout
# --------------------------------------------------------------------------- #
settings = render_sidebar()
lang = st.session_state.lang        # refresh in case the switcher changed it

utils.render_header(lang)
utils.render_disclaimer(lang)

tab_chat, tab_docs, tab_faq = st.tabs(
    [f"💬 {t('tab_chat', lang)}", f"📄 {t('tab_documents', lang)}",
     f"❓ {t('tab_faq', lang)}"]
)
with tab_chat:
    chat_tab(settings)
with tab_docs:
    documents_tab()
with tab_faq:
    faq_tab()

utils.render_footer(lang)
