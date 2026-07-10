"""
EchoMinutes — AI Meeting Assistant
Streamlit front end wired to the existing backend pipeline:
  utils.audio_process, core.transcriber, core.summary, core.extractor, core.rag_eng

UI: a ChatGPT-style shell — a collapsible dark sidebar (built entirely from our
own session_state, not Streamlit's native st.sidebar) plus a full-bleed main
area with a Studio panel (tabs) and a Chat panel. Dark theme only.

Why a custom sidebar instead of st.sidebar: Streamlit's built-in sidebar
toggle renders via Google's Material Symbols icon font. When that font can't
load (offline, blocked, corporate network) the control shows raw text like
"keyboard_double_arrow_right" instead of an icon, and its collapsed/expanded
state is also persisted in the browser's localStorage in a way that can
silently override anything we set from Python. Building the panel ourselves
with plain Unicode glyphs and a single session_state flag sidesteps both
problems entirely.
"""

import os
import html as html_lib
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from utils.audio_process import process_input
from core.transcriber import transcribe_all
from core.summary import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_eng import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="EchoMinutes — AI Meeting Assistant",
    page_icon="◈",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
_DEFAULTS = {
    "result": None,
    "chat_history": [],
    "processing": False,
    "sidebar_open": True,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# Display label -> internal value expected by core.transcriber
LANGUAGE_OPTIONS = {
    "English": "english",
    "Hindi / Bangla (Hinglish)": "hinglish",
}

# ----------------------------------------------------------------------------
# Color tokens — dark theme only
# ----------------------------------------------------------------------------
C = {
    "bg":          "#0B0B0D",
    "glow-1":      "rgba(232,163,61,0.07)",
    "glow-2":      "rgba(47,212,192,0.05)",
    "surface":     "#151517",
    "surface-2":   "#1C1C1F",
    "surface-3":   "#232326",
    "border":      "#2A2A2E",
    "border-soft": "#232326",
    "text":        "#ECECEE",
    "text-dim":    "#9B9BA3",
    "text-faint":  "#6C6C74",
    "accent":      "#E8A33D",
    "accent-ink":  "#17130A",
    "accent-soft": "rgba(232,163,61,0.12)",
    "wave":        "#2FD4C0",
    "accent-3":    "#B892E8",
    "danger":      "#E5695B",
    "success":     "#4CBB7D",
    "shadow":      "0 10px 30px rgba(0,0,0,0.4)",
    "shadow-sm":   "0 2px 10px rgba(0,0,0,0.3)",
}

# ----------------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------------
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {{
    --bg:          {C['bg']};
    --glow-1:      {C['glow-1']};
    --glow-2:      {C['glow-2']};
    --surface:     {C['surface']};
    --surface-2:   {C['surface-2']};
    --surface-3:   {C['surface-3']};
    --border:      {C['border']};
    --border-soft: {C['border-soft']};
    --text:        {C['text']};
    --text-dim:    {C['text-dim']};
    --text-faint:  {C['text-faint']};
    --accent:      {C['accent']};
    --accent-ink:  {C['accent-ink']};
    --accent-soft: {C['accent-soft']};
    --wave:        {C['wave']};
    --accent-3:    {C['accent-3']};
    --danger:      {C['danger']};
    --success:     {C['success']};
    --shadow:      {C['shadow']};
    --shadow-sm:   {C['shadow-sm']};
    --radius-lg:   16px;
    --radius-md:   10px;
    --radius-sm:   7px;
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
* {{ scroll-behavior: smooth; }}

html, body {{ background: var(--bg); margin: 0; padding: 0; }}

.stApp {{
    background:
        radial-gradient(circle at 12% -15%, var(--glow-1), transparent 32%),
        radial-gradient(circle at 100% 0%, var(--glow-2), transparent 28%),
        var(--bg);
    color: var(--text);
}}

/* Native Streamlit chrome we don't need — we build our own topbar/sidebar. */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ display: none; }}
[data-testid="stDeployButton"] {{ display: none !important; }}
[data-testid="stToolbarActions"] {{ display: none !important; }}
[data-testid="stStatusWidget"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}

/* ---- Full-bleed layout: no outer gap, edge-to-edge like a real app shell,
   and everything sized in rem/% so browser zoom never distorts it. Streamlit
   reserves top padding on these wrappers to make room for the header even
   though we've hidden it — zero them out explicitly. ---- */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main .block-container {{
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}}
/* Small breathing room on the left so content doesn't sit flush against the
   browser edge — everything else stays edge-to-edge. */
.main .block-container {{
    padding-left: 0.6rem !important;
}}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{ background: var(--bg); }}

/* ---- Clean, reliable secondary-button style — keyed off Streamlit's own
   `kind` attribute (not fragile structural selectors), so it never falls
   back to an unstyled black/blank button. ---- */
button[kind="secondary"] {{
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 8px !important;
    font-weight: 500 !important; box-shadow: none !important;
    transition: border-color 0.12s ease, color 0.12s ease;
}}
button[kind="secondary"]:hover:not(:disabled) {{
    border-color: var(--accent) !important; color: var(--accent) !important;
}}
button[kind="secondary"]:disabled {{
    opacity: 0.4 !important; color: var(--text-faint) !important;
}}
button[kind="primary"], .stButton > button:not([kind="secondary"]) {{
    background: var(--accent) !important; color: var(--accent-ink) !important; border: none !important;
    border-radius: var(--radius-sm) !important; font-weight: 600 !important;
    transition: transform 0.12s ease, opacity 0.12s ease;
}}
.stButton > button:not([kind="secondary"]):hover {{ transform: translateY(-1px); opacity: 0.92; }}
.stButton > button:not([kind="secondary"]):disabled {{ opacity: 0.4; transform: none; }}
.stButton > button {{ width: 100%; padding: 0.55rem 1.1rem; }}
.stDownloadButton > button {{
    background: transparent !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
}}
.stDownloadButton > button:hover {{ border-color: var(--accent) !important; color: var(--accent) !important; }}

a {{ color: var(--accent); }}
button:focus-visible, input:focus-visible, textarea:focus-visible {{
    outline: 2px solid var(--accent) !important; outline-offset: 2px;
}}

/* ---- Topbar (always full width, always in the same place) ---- */
.topbar-wrap {{ padding: 0.6rem 1.25rem; border-bottom: 1px solid var(--border-soft); }}
.topbar-left {{ display: flex; align-items: center; gap: 0.7rem; min-width: 0; padding-top: 0.15rem; }}
.hero-mark {{
    flex-shrink: 0; width: 34px; height: 34px; border-radius: 9px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--wave) 60%, var(--accent-3) 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif; font-size: 0.95rem; color: var(--accent-ink); font-weight: 700;
}}
.hero-title {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.15rem;
    letter-spacing: -0.01em; color: var(--text); margin: 0; line-height: 1.2;
}}
.hero-title em {{ color: var(--accent); font-style: normal; }}
.hero-sub {{ color: var(--text-faint); font-size: 0.74rem; margin: 0; }}
.status-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.03em;
    color: var(--text-dim); background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 999px; padding: 0.3rem 0.7rem; white-space: nowrap;
}}
.status-dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--text-faint); flex-shrink: 0; }}
.status-dot.on {{ background: var(--success); box-shadow: 0 0 0 3px var(--accent-soft); }}
.status-dot.busy {{ background: var(--accent); animation: pulse 1.2s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.35; }} }}

/* ---- The custom sidebar column — flush dark panel, full height feel, and
   a real sliding transition (both columns always stay mounted; only their
   size/opacity change, which is what lets the browser animate it smoothly
   instead of the whole layout jumping on every rerun). ---- */
div[data-testid="stHorizontalBlock"]:has(.side-marker) {{
    gap: 0 !important;
    align-items: stretch !important;
}}
div[data-testid="column"]:has(.side-marker),
div[data-testid="stColumn"]:has(.side-marker) {{
    background: var(--surface);
    border-right: 1px solid var(--border);
    padding: 1rem 1rem 1.25rem !important;
    min-height: calc(100vh - 3.2rem);
    overflow: hidden;
    opacity: 1;
    transition: flex-basis 0.28s cubic-bezier(0.4,0,0.2,1),
                width 0.28s cubic-bezier(0.4,0,0.2,1),
                min-width 0.28s cubic-bezier(0.4,0,0.2,1),
                padding 0.28s cubic-bezier(0.4,0,0.2,1),
                opacity 0.2s ease,
                border-color 0.2s ease;
}}
div[data-testid="column"]:has(.side-marker.closed),
div[data-testid="stColumn"]:has(.side-marker.closed) {{
    flex: 0 0 0% !important;
    width: 0 !important;
    min-width: 0 !important;
    padding: 0 !important;
    border-right-color: transparent;
    opacity: 0;
    pointer-events: none;
}}
div[data-testid="column"]:has(.main-marker),
div[data-testid="stColumn"]:has(.main-marker) {{
    padding: 1.5rem 2rem 2.5rem !important;
    transition: padding 0.28s cubic-bezier(0.4,0,0.2,1);
}}
@media (prefers-reduced-motion: reduce) {{
    div[data-testid="column"]:has(.side-marker),
    div[data-testid="stColumn"]:has(.side-marker),
    div[data-testid="column"]:has(.main-marker),
    div[data-testid="stColumn"]:has(.main-marker) {{
        transition: none !important;
    }}
}}
.side-brand {{
    display: flex; align-items: center; gap: 0.5rem;
    padding-bottom: 0.9rem; margin-bottom: 0.9rem;
    border-bottom: 1px solid var(--border);
}}
.side-mark {{
    width: 26px; height: 26px; border-radius: 7px; flex-shrink: 0;
    background: linear-gradient(135deg, var(--accent) 0%, var(--wave) 60%, var(--accent-3) 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif; font-size: 0.75rem; color: var(--accent-ink); font-weight: 700;
}}
.side-brand-text {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.02rem; color: var(--text);
}}
.side-brand-text em {{ color: var(--accent); font-style: normal; }}
.side-note {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.71rem;
    color: var(--text-faint); line-height: 1.8;
}}
.side-note .row {{ display: flex; align-items: center; gap: 0.4rem; }}

/* ---- Waveform loader ---- */
.waveform {{ display: flex; align-items: center; gap: 3px; height: 34px; margin: 0.4rem 0; }}
.waveform span {{
    display: block; width: 4px; border-radius: 2px; background: var(--accent);
    animation: wv 1.05s ease-in-out infinite;
}}
.waveform span:nth-child(1) {{ height: 10px; animation-delay: 0.0s; }}
.waveform span:nth-child(2) {{ height: 22px; animation-delay: 0.1s; }}
.waveform span:nth-child(3) {{ height: 32px; animation-delay: 0.2s; }}
.waveform span:nth-child(4) {{ height: 16px; animation-delay: 0.3s; }}
.waveform span:nth-child(5) {{ height: 26px; animation-delay: 0.4s; }}
.waveform span:nth-child(6) {{ height: 14px; animation-delay: 0.5s; }}
.waveform span:nth-child(7) {{ height: 30px; animation-delay: 0.6s; }}
.waveform span:nth-child(8) {{ height: 18px; animation-delay: 0.7s; }}
@keyframes wv {{
    0%, 100% {{ transform: scaleY(0.4); opacity: 0.6; }}
    50%      {{ transform: scaleY(1);   opacity: 1;   }}
}}
@media (prefers-reduced-motion: reduce) {{
    .waveform span {{ animation: none; transform: scaleY(0.7); opacity: 0.9; }}
    .status-dot.busy {{ animation: none; }}
}}

/* ---- Cards ---- */
.card {{
    background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
    padding: 1.3rem 1.45rem; margin-bottom: 1rem; box-shadow: var(--shadow-sm);
    word-wrap: break-word; overflow-wrap: break-word;
}}
.card-empty {{
    text-align: center; color: var(--text-faint); font-size: 0.86rem;
    padding: 1.6rem 1rem; font-style: italic;
}}
.eyebrow {{
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--accent); margin-bottom: 0.6rem;
}}
.stat-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
    color: var(--text-dim); background: var(--surface-2); border: 1px solid var(--border);
    border-radius: 999px; padding: 0.25rem 0.7rem; margin: 0.15rem 0.4rem 0.15rem 0;
}}
.dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--wave); display: inline-block; flex-shrink:0; }}

.meeting-title {{
    font-family: 'Fraunces', serif; font-size: clamp(1.1rem, 1.8vw, 1.4rem); font-weight: 600;
    margin: 0 0 0.6rem 0; overflow: hidden; text-overflow: ellipsis;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}}

/* ---- Pane headers (Studio / Chat) ---- */
.pane-header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.7rem; }}
.pane-title {{ font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.02rem; color: var(--text); }}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; border-bottom: 1px solid var(--border);
    overflow-x: auto; flex-wrap: nowrap; scrollbar-width: thin;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif; font-weight: 500; color: var(--text-dim);
    padding: 0.6rem 1rem; white-space: nowrap;
}}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--accent) !important; }}

/* ---- Inputs ---- */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {{
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif;
}}
.stTextInput input:focus {{ border-color: var(--accent) !important; }}
.stFileUploader > div {{
    background: var(--surface-2); border: 1px dashed var(--border); border-radius: var(--radius-md);
}}
.stFileUploader section {{ background: transparent; }}
.stRadio > div {{ gap: 0.4rem; }}
.stRadio label {{ font-size: 0.86rem; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color: var(--text-faint) !important; }}

/* ---- Transcript block ---- */
.transcript-block {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; line-height: 1.75;
    color: var(--text-dim); white-space: pre-wrap;
    max-height: 560px; overflow-y: auto; padding-right: 0.3rem;
}}

/* ---- Chat ---- */
.stChatMessage {{ background: var(--surface-2); border: 1px solid var(--border); border-radius: var(--radius-md); }}
div[data-testid="stChatInput"] textarea {{
    background: var(--surface-2) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important;
}}
div[data-testid="stChatInput"] {{ border-top: 1px solid var(--border-soft); padding-top: 0.6rem; }}

/* ---- Empty states ---- */
.empty-hero {{ text-align: center; padding: 3.2rem 1.5rem 2.6rem; }}
.empty-hero .glyph {{
    width: 54px; height: 54px; margin: 0 auto 1rem; border-radius: 14px;
    background: var(--surface-2); border: 1px solid var(--border);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif; font-size: 1.4rem; color: var(--accent);
}}

/* ---- How-it-works step strip ---- */
.step-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;
    max-width: 780px; margin: 1.8rem auto 0; text-align: left;
}}
.step-card {{
    background: var(--surface-2); border: 1px solid var(--border-soft); border-radius: var(--radius-md);
    padding: 1.1rem 1.15rem;
}}
.step-num {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: var(--accent-soft); color: var(--accent);
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 600;
    margin-bottom: 0.6rem;
}}
.step-title {{ font-weight: 600; font-size: 0.88rem; color: var(--text); margin-bottom: 0.25rem; }}
.step-desc {{ font-size: 0.79rem; color: var(--text-dim); line-height: 1.5; }}
@media (max-width: 768px) {{ .step-grid {{ grid-template-columns: 1fr; }} }}

.hr {{ height: 1px; background: var(--border); margin: 1.4rem 0; border: none; }}

/* ---- Scrollbars ---- */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ---- Responsive: stack the studio/chat pane under ~900px ---- */
@media (max-width: 900px) {{
    div[data-testid="stHorizontalBlock"]:has(.pane-marker) {{ flex-wrap: wrap !important; }}
    div[data-testid="stHorizontalBlock"]:has(.pane-marker) > div[data-testid="column"],
    div[data-testid="stHorizontalBlock"]:has(.pane-marker) > div[data-testid="stColumn"] {{
        width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
    }}
}}
@media (max-width: 768px) {{
    div[data-testid="column"]:has(.main-marker),
    div[data-testid="stColumn"]:has(.main-marker) {{ padding: 1.1rem !important; }}
    .card {{ padding: 1.05rem; }}
}}
</style>
""", unsafe_allow_html=True)


def reset_session():
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.processing = False
    for widget_key in ("source_url", "source_file", "lang_select"):
        st.session_state.pop(widget_key, None)


# ----------------------------------------------------------------------------
# Custom sidebar (Sources) — plain Python/session_state, no st.sidebar
# ----------------------------------------------------------------------------
def render_sources_panel():
    st.markdown(
        '<div class="side-brand">'
        '<span class="side-mark">EM</span>'
        '<span class="side-brand-text">Echo<em>Minutes</em></span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.button("＋  New session", key="new_session_btn", type="secondary", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown('<div class="eyebrow" style="margin-top:1.1rem;">Source</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input type", ["YouTube URL", "Upload file"],
        label_visibility="collapsed", horizontal=True,
    )

    src = None

    if input_mode == "YouTube URL":
        src = st.text_input(
            "YouTube URL", placeholder="https://youtu.be/…",
            label_visibility="collapsed", key="source_url",
        ).strip()
        if src and not (src.startswith("http://") or src.startswith("https://")):
            st.caption("⚠ That doesn't look like a valid URL.")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio or video", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"],
            label_visibility="collapsed", key="source_file",
        )
        if uploaded_file is not None:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > 500:
                st.caption(f"⚠ File is {size_mb:.0f} MB — large files take longer to process.")
            os.makedirs("downloads", exist_ok=True)
            temp_path = f"downloads/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            src = temp_path
            st.caption(f"Ready: {uploaded_file.name} ({size_mb:.1f} MB)")

    st.markdown('<div class="eyebrow" style="margin-top:1.1rem;">Language</div>', unsafe_allow_html=True)
    lang_label = st.selectbox(
        "Language", list(LANGUAGE_OPTIONS.keys()),
        label_visibility="collapsed", key="lang_select",
    )
    lang = LANGUAGE_OPTIONS[lang_label]

    st.markdown('<hr class="hr">', unsafe_allow_html=True)

    run_clicked = st.button(
        "▶  Run analysis",
        use_container_width=True,
        disabled=st.session_state.processing or not src,
    )
    if st.session_state.processing:
        st.caption("Processing… this tab will update automatically.")

    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown(
        '<div class="side-note">'
        '<div class="row">◦ Local Whisper transcription</div>'
        '<div class="row">◦ Hindi / Bangla → English</div>'
        '<div class="row">◦ LangChain + Mistral summarization</div>'
        '<div class="row">◦ ChromaDB retrieval chat</div>'
        '</div>', unsafe_allow_html=True,
    )

    return src, lang, lang_label, run_clicked


# ----------------------------------------------------------------------------
# Topbar — one reliable Unicode toggle, brand, status. Always the same spot.
# ----------------------------------------------------------------------------
st.markdown('<div class="topbar-wrap">', unsafe_allow_html=True)
top_toggle, top_brand, top_status = st.columns([0.5, 6.5, 1.4])

with top_toggle:
    toggle_icon = "‹" if st.session_state.sidebar_open else "☰"
    if st.button(toggle_icon, key="sidebar_toggle", type="secondary",
                 help="Show or hide the sidebar"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open
        st.rerun()

with top_brand:
    st.markdown("""
    <div class="topbar-left">
        <div class="hero-mark">EM</div>
        <div>
            <p class="hero-title">Echo<em>Minutes</em></p>
            <p class="hero-sub">Turn any recorded meeting into a transcript, a summary, and a conversation.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with top_status:
    if st.session_state.processing:
        dot, label = "busy", "Processing"
    elif st.session_state.result is not None:
        dot, label = "on", "Ready"
    else:
        dot, label = "", "Idle"
    st.markdown(
        f'<div style="padding-top:0.35rem; text-align:right;"><span class="status-pill">'
        f'<span class="status-dot {dot}"></span>{label}</span></div>',
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Body — sidebar + main column. Both columns always stay mounted; the
# sidebar's size/opacity just animate to 0 when closed (see CSS above), which
# is what gives it a real sliding transition instead of an instant layout
# jump on every rerun.
# ----------------------------------------------------------------------------
side_col, main_col = st.columns([1, 3.3], gap="small")

with side_col:
    state_class = "open" if st.session_state.sidebar_open else "closed"
    st.markdown(f'<span class="side-marker {state_class}"></span>', unsafe_allow_html=True)
    if st.session_state.sidebar_open:
        source, language, language_label, run = render_sources_panel()
    else:
        source, language, language_label, run = None, "english", "English", False

with main_col:
    st.markdown('<span class="main-marker"></span>', unsafe_allow_html=True)

    # ------------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------------
    def run_pipeline(src: str, lang: str) -> dict:
        chunks = process_input(src)
        transcript = transcribe_all(chunks, lang)
        title = generate_title(transcript)
        summary = summarize(transcript)
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        rag_chain = build_rag_chain(transcript)
        return {
            "title": title or "Untitled meeting",
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
            "language_label": language_label,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    if run:
        if not source:
            st.warning("Add a YouTube URL or upload a file before running.")
        else:
            st.session_state.processing = True
            st.session_state.chat_history = []

            stage = st.empty()
            stage.markdown("""
            <div class="card">
                <div class="waveform">
                    <span></span><span></span><span></span><span></span>
                    <span></span><span></span><span></span><span></span>
                </div>
                <div style="color:var(--text-dim); font-family:'JetBrains Mono',monospace; font-size:0.82rem;">
                    Processing — this can take a few minutes on CPU, depending on audio length.
                </div>
            </div>
            """, unsafe_allow_html=True)

            try:
                result = run_pipeline(source, language)
                st.session_state.result = result
            except Exception as e:
                st.session_state.result = None
                st.session_state.processing = False
                stage.empty()
                st.error(f"Something went wrong while processing this meeting: {e}")
                st.stop()
            finally:
                st.session_state.processing = False
            stage.empty()
            st.rerun()

    # ------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------
    def render_or_empty(content: str, empty_label: str):
        if content and str(content).strip():
            st.markdown(f'<div class="card">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="card card-empty">{empty_label}</div>', unsafe_allow_html=True)

    result = st.session_state.result

    if result is None:
        st.markdown("""
        <div class="card empty-hero">
            <div class="glyph">◈</div>
            <div class="eyebrow" style="justify-content:center;">Waiting for input</div>
            <p style="color:var(--text-dim); max-width:480px; margin:0.5rem auto 0; line-height:1.6;">
                Paste a YouTube link or upload a recording in the sidebar, then run the analysis.
                EchoMinutes will transcribe it locally, translate Hindi or Bangla speech if needed,
                and generate a summary you can chat with — just like a notebook for your meetings.
            </p>
            <div class="step-grid">
                <div class="step-card">
                    <div class="step-num">1</div>
                    <div class="step-title">Add a source</div>
                    <div class="step-desc">Paste a YouTube link or upload an audio/video file in the Sources panel on the left.</div>
                </div>
                <div class="step-card">
                    <div class="step-num">2</div>
                    <div class="step-title">Run analysis</div>
                    <div class="step-desc">Local Whisper (or Sarvam AI for Hindi/Bangla) transcribes it, then Mistral summarizes it.</div>
                </div>
                <div class="step-card">
                    <div class="step-num">3</div>
                    <div class="step-title">Explore &amp; chat</div>
                    <div class="step-desc">Review the summary, action items, and decisions, or ask the transcript a question directly.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        word_count = len(result["transcript"].split()) if result["transcript"] else 0
        st.markdown(f"""
        <div class="card">
            <div class="eyebrow">Meeting</div>
            <p class="meeting-title">{html_lib.escape(result['title'])}</p>
            <span class="stat-pill"><span class="dot"></span>Generated {result['generated_at']}</span>
            <span class="stat-pill"><span class="dot"></span>{html_lib.escape(result.get('language_label', ''))}</span>
            <span class="stat-pill"><span class="dot"></span>{word_count:,} words transcribed</span>
        </div>
        """, unsafe_allow_html=True)

        # ---- Two-pane layout: Studio (outputs) + Chat ----
        st.markdown('<span class="pane-marker"></span>', unsafe_allow_html=True)
        col_studio, col_chat = st.columns([1.15, 1], gap="large")

        with col_studio:
            st.markdown(
                '<div class="pane-header"><span class="pane-title">Studio</span></div>',
                unsafe_allow_html=True,
            )
            tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript = st.tabs(
                ["Summary", "Action Items", "Key Decisions", "Open Questions", "Transcript"]
            )

            with tab_summary:
                render_or_empty(result["summary"], "No summary was generated for this meeting.")

            with tab_actions:
                render_or_empty(result["action_items"], "No action items were found in this meeting.")

            with tab_decisions:
                render_or_empty(result["key_decisions"], "No key decisions were identified.")

            with tab_questions:
                render_or_empty(result["open_questions"], "No open questions were identified.")

            with tab_transcript:
                if result["transcript"] and result["transcript"].strip():
                    st.markdown(
                        f'<div class="card"><div class="transcript-block">'
                        f'{html_lib.escape(result["transcript"])}</div></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown('<div class="card card-empty">Transcript is empty.</div>', unsafe_allow_html=True)

        with col_chat:
            st.markdown(
                '<div class="pane-header"><span class="pane-title">Chat</span></div>',
                unsafe_allow_html=True,
            )

            chat_box = st.container(height=460, border=False)
            with chat_box:
                if not st.session_state.chat_history:
                    st.markdown(
                        '<div class="card card-empty">Ask a question about the transcript to get started.</div>',
                        unsafe_allow_html=True,
                    )
                for role, msg in st.session_state.chat_history:
                    with st.chat_message(role):
                        st.write(msg)

            question = st.chat_input("Ask a question about the transcript…")
            if question and question.strip():
                st.session_state.chat_history.append(("user", question))
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"Couldn't answer that: {e}"
                st.session_state.chat_history.append(("assistant", answer))
                st.rerun()

        # ---- Export ----
        st.markdown('<hr class="hr">', unsafe_allow_html=True)
        st.markdown('<div class="eyebrow">Export</div>', unsafe_allow_html=True)

        export_text = (
            f"EchoMinutes — {result['title']}\n"
            f"Generated: {result['generated_at']}\n"
            f"{'=' * 60}\n\n"
            f"SUMMARY\n{result['summary'] or '—'}\n\n"
            f"ACTION ITEMS\n{result['action_items'] or '—'}\n\n"
            f"KEY DECISIONS\n{result['key_decisions'] or '—'}\n\n"
            f"OPEN QUESTIONS\n{result['open_questions'] or '—'}\n\n"
            f"FULL TRANSCRIPT\n{result['transcript'] or '—'}\n"
        )
        safe_name = "".join(c for c in result["title"][:40] if c.isalnum() or c in " _-").strip().replace(" ", "_") or "meeting"

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download as TXT", data=export_text,
                file_name=f"{safe_name}.txt", mime="text/plain", use_container_width=True,
            )
        with col2:
            try:
                from fpdf import FPDF

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", size=11)
                for line in export_text.split("\n"):
                    safe_line = line.encode("latin-1", "replace").decode("latin-1")
                    pdf.multi_cell(0, 6, safe_line)
                pdf_bytes = bytes(pdf.output(dest="S"))

                st.download_button(
                    "Download as PDF", data=pdf_bytes,
                    file_name=f"{safe_name}.pdf", mime="application/pdf", use_container_width=True,
                )
            except Exception:
                st.button("PDF export unavailable", disabled=True, use_container_width=True)
