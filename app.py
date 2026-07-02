"""
EchoMinutes — AI Meeting Assistant
Streamlit front end wired to the existing backend pipeline:
  utils.audio_process, core.transcriber, core.summary, core.extractor, core.rag_eng
"""

import os
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
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theme state
# ----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "night"
if "result" not in st.session_state:
    st.session_state.result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False

THEMES = {
    "night": {
        "bg":        "#0B0B0D",
        "bg-glow-1": "rgba(232,163,61,0.07)",
        "bg-glow-2": "rgba(47,212,192,0.05)",
        "panel":     "#151517",
        "panel-2":   "#1C1C1F",
        "line":      "#2A2A2E",
        "text":      "#ECECEE",
        "text-dim":  "#9B9BA3",
        "signal":    "#E8A33D",
        "signal-on": "#17130A",
        "wave":      "#2FD4C0",
        "accent-3":  "#B892E8",
        "danger":    "#E5695B",
        "shadow":    "0 8px 28px rgba(0,0,0,0.45)",
    },
    "day": {
        "bg":        "#FFFFFF",
        "bg-glow-1": "rgba(201,124,29,0.05)",
        "bg-glow-2": "rgba(18,150,138,0.04)",
        "panel":     "#F7F7F8",
        "panel-2":   "#F0F0F2",
        "line":      "#E3E3E7",
        "text":      "#1B1B1F",
        "text-dim":  "#6B6B74",
        "signal":    "#C97C1D",
        "signal-on": "#FFFFFF",
        "wave":      "#12968A",
        "accent-3":  "#8256C9",
        "danger":    "#C4453B",
        "shadow":    "0 8px 24px rgba(30,40,70,0.06)",
    },
}
T = THEMES[st.session_state.theme]

# ----------------------------------------------------------------------------
# Design system
# ----------------------------------------------------------------------------
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {{
    --bg:        {T['bg']};
    --glow-1:    {T['bg-glow-1']};
    --glow-2:    {T['bg-glow-2']};
    --panel:     {T['panel']};
    --panel-2:   {T['panel-2']};
    --line:      {T['line']};
    --text:      {T['text']};
    --text-dim:  {T['text-dim']};
    --signal:    {T['signal']};
    --signal-on: {T['signal-on']};
    --wave:      {T['wave']};
    --accent-3:  {T['accent-3']};
    --danger:    {T['danger']};
    --shadow:    {T['shadow']};
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background:
        radial-gradient(circle at 12% -15%, var(--glow-1), transparent 32%),
        radial-gradient(circle at 100% 0%, var(--glow-2), transparent 28%),
        var(--bg);
    color: var(--text);
    transition: background 0.25s ease, color 0.25s ease;
}}

#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stDeployButton"] {{ display: none !important; }}

.main .block-container {{
    padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1180px;
    margin-left: auto; margin-right: auto;
}}

/* ---- Hero ---- */
.hero {{
    display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    flex-wrap: wrap;
    padding: 1.6rem 1.8rem; margin-bottom: 1.6rem;
    background: linear-gradient(135deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--shadow);
}}
.hero-title {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: clamp(1.5rem, 2.4vw, 2.1rem);
    letter-spacing: -0.01em; color: var(--text); margin: 0;
}}
.hero-title em {{ color: var(--signal); font-style: normal; }}
.hero-sub {{ color: var(--text-dim); font-size: 0.9rem; margin-top: 0.35rem; }}
.hero-mark {{
    flex-shrink: 0;
    width: 46px; height: 46px; border-radius: 10px;
    background: linear-gradient(135deg, var(--signal) 0%, var(--wave) 60%, var(--accent-3) 100%);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif; font-size: 1.3rem; color: var(--signal-on); font-weight: 700;
}}

/* ---- Waveform loader ---- */
.waveform {{ display: flex; align-items: center; gap: 3px; height: 34px; margin: 0.4rem 0; }}
.waveform span {{
    display: block; width: 4px; border-radius: 2px; background: var(--signal);
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
}}

/* ---- Cards ---- */
.card {{
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 1.3rem 1.45rem; margin-bottom: 1rem; box-shadow: var(--shadow);
    word-wrap: break-word; overflow-wrap: break-word;
}}
.card-empty {{
    text-align: center; color: var(--text-dim); font-size: 0.88rem;
    padding: 1.4rem 1rem; font-style: italic;
}}
.eyebrow {{
    display: flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--signal); margin-bottom: 0.5rem;
}}
.stat-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: 'JetBrains Mono', monospace; font-size: 0.76rem;
    color: var(--text-dim); background: var(--panel-2); border: 1px solid var(--line);
    border-radius: 999px; padding: 0.25rem 0.7rem; margin: 0.15rem 0.4rem 0.15rem 0;
}}
.dot {{ width: 6px; height: 6px; border-radius: 50%; background: var(--wave); display: inline-block; flex-shrink:0; }}

.meeting-title {{
    font-family: 'Fraunces', serif; font-size: clamp(1.15rem, 2vw, 1.5rem); font-weight: 600;
    margin: 0 0 0.6rem 0; overflow: hidden; text-overflow: ellipsis;
    display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical;
}}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: var(--panel);
    border-right: 1px solid var(--line);
    min-width: 300px !important;
}}
section[data-testid="stSidebar"] > div {{ padding-top: 0; }}
.sidebar-header {{
    position: sticky; top: 0; z-index: 5;
    background: var(--panel);
    display: flex; align-items: center; justify-content: space-between;
    padding: 1.1rem 0.2rem 0.9rem 0.2rem;
    border-bottom: 1px solid var(--line); margin-bottom: 1.1rem;
}}
.sidebar-brand {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 1.05rem; color: var(--text);
}}
.sidebar-brand em {{ color: var(--signal); font-style: normal; }}

/* ---- Theme toggle button ---- */
.theme-btn button {{
    background: var(--panel-2) !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 8px !important;
    padding: 0.35rem 0.6rem !important; font-size: 0.85rem !important;
    box-shadow: none !important;
}}

/* ---- Buttons ---- */
.stButton > button {{
    background: var(--signal); color: var(--signal-on); border: none; border-radius: 8px;
    font-weight: 600; padding: 0.55rem 1.1rem;
    transition: transform 0.12s ease, opacity 0.12s ease; width: 100%;
}}
.stButton > button:hover {{ transform: translateY(-1px); opacity: 0.92; }}
.stButton > button:disabled {{ opacity: 0.45; transform: none; }}
.stDownloadButton > button {{
    background: transparent !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 8px !important;
}}

/* ---- Tabs (horizontally scrollable on small screens) ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; border-bottom: 1px solid var(--line);
    overflow-x: auto; flex-wrap: nowrap; scrollbar-width: thin;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'Inter', sans-serif; font-weight: 500; color: var(--text-dim);
    padding: 0.6rem 1rem; white-space: nowrap;
}}
.stTabs [aria-selected="true"] {{ color: var(--signal) !important; }}

/* ---- Inputs ---- */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div {{
    background: var(--panel-2) !important; color: var(--text) !important;
    border: 1px solid var(--line) !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif;
}}
.stFileUploader > div {{
    background: var(--panel-2); border: 1px dashed var(--line); border-radius: 10px;
}}
.stRadio > div {{ gap: 0.4rem; }}

/* ---- Transcript block ---- */
.transcript-block {{
    font-family: 'JetBrains Mono', monospace; font-size: 0.83rem; line-height: 1.7;
    color: var(--text-dim); white-space: pre-wrap;
    max-height: 480px; overflow-y: auto; padding-right: 0.3rem;
}}

/* ---- Chat ---- */
.stChatMessage {{ background: var(--panel-2); border: 1px solid var(--line); border-radius: 10px; }}

.hr {{ height: 1px; background: var(--line); margin: 1.4rem 0; border: none; }}

/* ---- Scrollbars ---- */
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-thumb {{ background: var(--line); border-radius: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}

/* ---- Responsive ---- */
@media (max-width: 768px) {{
    .block-container {{ padding-left: 0.9rem; padding-right: 0.9rem; }}
    .hero {{ padding: 1.2rem; flex-direction: column; align-items: flex-start; }}
    .hero-mark {{ order: -1; margin-bottom: 0.4rem; }}
    .card {{ padding: 1.05rem; }}
    section[data-testid="stSidebar"] {{ min-width: 260px !important; }}
}}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown(
            '<div class="sidebar-header"><span class="sidebar-brand">Echo<em>Minutes</em></span></div>',
            unsafe_allow_html=True,
        )
    with header_col2:
        st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
        icon = "☾" if st.session_state.theme == "night" else "☀"
        if st.button(icon, key="theme_toggle", help="Toggle day / night mode"):
            st.session_state.theme = "day" if st.session_state.theme == "night" else "night"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Source</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input type", ["YouTube URL", "Upload file"],
        label_visibility="collapsed", horizontal=True,
    )

    source = None

    if input_mode == "YouTube URL":
        source = st.text_input(
            "YouTube URL", placeholder="https://youtu.be/…", label_visibility="collapsed"
        ).strip()
        if source and not (source.startswith("http://") or source.startswith("https://")):
            st.caption("⚠ That doesn't look like a valid URL.")
    else:
        uploaded_file = st.file_uploader(
            "Upload audio or video", type=["mp3", "wav", "m4a", "mp4", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > 500:
                st.caption(f"⚠ File is {size_mb:.0f} MB — large files take longer to process.")
            os.makedirs("downloads", exist_ok=True)
            temp_path = f"downloads/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source = temp_path
            st.caption(f"Ready: {uploaded_file.name} ({size_mb:.1f} MB)")

    st.markdown('<div class="eyebrow" style="margin-top:1.3rem;">Language</div>', unsafe_allow_html=True)
    language = st.selectbox("Language", ["english", "hinglish"], label_visibility="collapsed")

    st.markdown('<hr class="hr">', unsafe_allow_html=True)

    run = st.button(
        "▶  Run analysis",
        use_container_width=True,
        disabled=st.session_state.processing or not source,
    )
    if st.session_state.processing:
        st.caption("Processing… this tab will update automatically.")

    st.markdown('<hr class="hr">', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; '
        'color:var(--text-dim); line-height:1.7;">'
        'Local Whisper transcription<br>Hindi / Bangla → English<br>'
        'LangChain + Mistral summarization<br>ChromaDB retrieval chat'
        '</div>', unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div>
        <p class="hero-title">Echo<em>Minutes</em></p>
        <p class="hero-sub">Turn any recorded meeting into a transcript, a summary, and a conversation.</p>
    </div>
    <div class="hero-mark">EM</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_pipeline(source: str, language: str) -> dict:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
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


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def render_or_empty(content: str, empty_label: str):
    if content and str(content).strip():
        st.markdown(f'<div class="card">{content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card card-empty">{empty_label}</div>', unsafe_allow_html=True)


result = st.session_state.result

if result is None:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem 1.25rem;">
        <div class="eyebrow" style="justify-content:center;">Waiting for input</div>
        <p style="color:var(--text-dim); max-width:480px; margin:0.5rem auto 0; line-height:1.6;">
            Paste a YouTube link or upload a recording in the sidebar, then run the analysis.
            EchoMinutes will transcribe it locally, translate Hindi or Bangla speech if needed,
            and generate a summary you can chat with.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="card">
        <div class="eyebrow">Meeting</div>
        <p class="meeting-title">{result['title']}</p>
        <span class="stat-pill"><span class="dot"></span>Generated {result['generated_at']}</span>
        <span class="stat-pill"><span class="dot"></span>{language}</span>
    </div>
    """, unsafe_allow_html=True)

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["Summary", "Action Items", "Key Decisions", "Open Questions", "Transcript", "Chat"]
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
                f'<div class="card"><div class="transcript-block">{result["transcript"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="card card-empty">Transcript is empty.</div>', unsafe_allow_html=True)

    with tab_chat:
        st.markdown('<div class="eyebrow">Ask about this meeting</div>', unsafe_allow_html=True)

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
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        answer = ask_question(result["rag_chain"], question)
                    except Exception as e:
                        answer = f"Couldn't answer that: {e}"
                st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))

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