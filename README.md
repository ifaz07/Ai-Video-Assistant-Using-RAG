# EchoMinutes — AI Meeting Assistant

**EchoMinutes** turns any recorded meeting — a YouTube link or a local audio/video file — into a clean transcript, an executive summary, structured action items, key decisions, open questions, and a chat interface you can interrogate for details. Everything runs through a local Whisper model plus a Mistral-powered LangChain pipeline, with a Streamlit UI styled as a NotebookLM-style workspace with full light/dark mode support.

I built this to stop losing meeting context to my own notes. Paste a link or drop a file in, and a few minutes later I have a searchable record of what was actually said and decided.

---

## What it does

- **Transcription** — Local Whisper transcribes English audio on CPU/GPU; Hindi/Bangla ("Hinglish") audio is routed to Sarvam AI's speech-to-text-translate API, which transcribes and translates to English in one pass.
- **Summarization** — A map-reduce chain (LangChain + Mistral) chunks the transcript, summarizes each piece, then merges those into one professional bullet-point summary.
- **Extraction** — Separate LLM passes pull out action items (with owner and deadline where mentioned), key decisions, and open questions.
- **Chat (RAG)** — The transcript is embedded (`sentence-transformers/all-MiniLM-L6-v2`) and stored in ChromaDB, so you can ask follow-up questions and get answers grounded strictly in the transcript, with a fallback response when the answer isn't in scope.
- **Export** — Download the full session (summary, action items, decisions, questions, transcript) as `.txt` or `.pdf`.
- **UI** — A Streamlit front end with a Sources sidebar, a Studio panel (tabs for Summary / Action Items / Key Decisions / Open Questions / Transcript), a persistent chat pane, and a topbar toggle for dark/light mode.

## How it works

```
Source (YouTube URL or uploaded file)
        │
        ▼
utils/audio_process.py   → download / convert to WAV, split into 10-minute chunks
        │
        ▼
core/transcriber.py      → Whisper (English) or Sarvam AI (Hindi/Bangla → English)
        │
        ▼
core/summary.py          → map-reduce summarization + title generation (Mistral)
core/extractor.py        → action items / key decisions / open questions (Mistral)
core/vectorDB.py         → chunk + embed transcript into ChromaDB
core/rag_eng.py          → retrieval-augmented chat over the transcript (Mistral)
        │
        ▼
app.py                   → Streamlit UI: sidebar, results tabs, chat, export
```

## Tech stack

| Layer | Tools |
|---|---|
| UI | Streamlit |
| Speech-to-text | OpenAI Whisper (local), Sarvam AI STT-translate API |
| LLM orchestration | LangChain (LCEL) |
| LLM | Mistral (`mistral-small-latest`) via `langchain-mistralai` |
| Embeddings / vector store | `sentence-transformers` (all-MiniLM-L6-v2) + ChromaDB |
| Audio handling | `pydub`, `ffmpeg`, `yt-dlp` |
| Export | `fpdf2` |

## Project structure

```
.
├── app.py                  # Streamlit UI (only file you should need to edit for UI changes)
├── main.py                 # CLI / script entry point
├── core/
│   ├── transcriber.py       # Whisper + Sarvam AI transcription
│   ├── summary.py           # Map-reduce summarization + title generation
│   ├── extractor.py         # Action items / decisions / questions extraction
│   ├── rag_eng.py           # RAG chat chain over the transcript
│   └── vectorDB.py          # ChromaDB embedding + retrieval
├── utils/
│   └── audio_process.py     # Download, convert, and chunk audio
├── downloads/                # Downloaded/uploaded audio (gitignored)
├── vector_db/                 # Persisted ChromaDB store (gitignored)
├── Requirements.txt
└── .env                        # API keys and model config (gitignored)
```

## Getting started

### Prerequisites

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) installed and on your `PATH` (required by `pydub` and `yt-dlp`)
- A [Mistral AI](https://mistral.ai/) API key
- A [Sarvam AI](https://www.sarvam.ai/) API key (only needed for Hindi/Bangla transcription)

### Installation

```bash
git clone <your-repo-url>
cd "AI Video Assistant"

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r Requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
WHISPER_MODEL=small
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5
```

`WHISPER_MODEL` accepts any standard Whisper model size (`tiny`, `base`, `small`, `medium`, `large`) — larger models are more accurate but slower on CPU.

### Run

```bash
streamlit run app.py
```

Then, in the app:

1. Paste a YouTube URL or upload an audio/video file in the **Source** panel.
2. Choose the language — **English** (Whisper) or **Hindi / Bangla (Hinglish)** (Sarvam AI).
3. Click **Run analysis** and wait for transcription + summarization to complete.
4. Review the summary, action items, decisions, and open questions in the **Studio** panel, chat with the transcript in the **Chat** panel, and export the results as TXT or PDF.

## Notes & limitations

- Whisper runs locally and on CPU by default, so processing time scales with recording length — expect a few minutes for a typical meeting.
- Large uploads (500MB+) will take noticeably longer to process; the UI will flag this.
- The RAG chat is intentionally scoped to the transcript — it will say so explicitly if an answer isn't supported by the meeting content, rather than guessing.
- `.env`, `downloads/`, and `vector_db/` are gitignored; don't commit API keys or generated audio/vector data.

## Roadmap

- [ ] Persistent history across sessions (multiple past meetings, not just the current one)
- [ ] Speaker diarization
- [ ] Support for additional languages beyond English and Hindi/Bangla
- [ ] Dockerized deployment

## License

Add a license of your choice here (MIT, Apache 2.0, etc.).
