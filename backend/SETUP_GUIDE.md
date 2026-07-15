# STEK 2035 Chatbot — Full Setup (Next.js + Local RAG + Ollama)

Your architecture has three pieces:

```
Browser (Next.js UI)
       │  POST /api/chat
       ▼
Next.js API route  →  proxies to  →  Python FastAPI (rag_server.py)
                                            │
                                            ├─ 1. Embeds your question (multilingual-e5-base)
                                            ├─ 2. Finds top-5 matching chunks (your embeddings.npy)
                                            ├─ 3. Builds a prompt with that context
                                            └─ 4. Sends prompt to Ollama (local LLM) → gets answer
```

---

## Part 1 — Install and start Ollama

1. Download Ollama: https://ollama.com/download
2. Pull a model (pick one that fits your hardware — you have 20GB RAM, no dedicated GPU, so a mid-size model works best):
   ```bash
   ollama pull llama3.1
   ```
   Other good options: `mistral`, `qwen2.5:7b` (qwen handles German well).
3. Ollama runs automatically as a background service on `http://localhost:11434`. Verify:
   ```bash
   ollama list
   ```

If you want to use a different model, edit `OLLAMA_MODEL` in `rag_server.py`.

---

## Part 2 — Set up the Python RAG backend

1. Copy the `backend/` folder (from this response) into your `STEK2035-chatbot-main` project root, so it sits next to your existing `vector_store/` folder:

   ```
   STEK2035-chatbot-main/
   ├── vector_store/
   │   ├── chunks.jsonl
   │   ├── embeddings.npy
   │   └── meta.json
   ├── backend/
   │   ├── rag_server.py
   │   └── requirements.txt
   └── ... (your existing folders)
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   cd STEK2035-chatbot-main
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r backend/requirements.txt
   ```

3. **Important**: `rag_server.py` expects `vector_store/` to be in its working directory. Run it from the project root:
   ```bash
   cd STEK2035-chatbot-main
   uvicorn backend.rag_server:app --reload --port 8000
   ```
   The first run downloads the `multilingual-e5-base` model (~1GB) — this only happens once.

4. Verify it's running:
   ```bash
   curl http://localhost:8000/health
   ```
   You should see `{"status":"ok","chunks_loaded":1307,...}`

---

## Part 3 — Set up the Next.js frontend

1. Create the Next.js project (skip if already done):
   ```bash
   npx create-next-app@latest stek-chatbot-frontend --typescript
   cd stek-chatbot-frontend
   ```

2. Replace `app/page.tsx` with the provided `frontend/page.tsx`.

3. Create `app/api/chat/route.ts` and paste in the provided `frontend/route.ts`.

4. (Optional) Create `.env.local` if your backend runs on a different port/host:
   ```
   RAG_BACKEND_URL=http://localhost:8000/chat
   ```

5. Start the frontend:
   ```bash
   npm run dev
   ```

6. Open **http://localhost:3000**

---

## Running everything together

You need **3 terminals** open simultaneously:

| Terminal | Command | Purpose |
|----------|---------|---------|
| 1 | `ollama serve` (usually auto-running) | Local LLM |
| 2 | `uvicorn backend.rag_server:app --port 8000` | RAG retrieval + generation |
| 3 | `npm run dev` | Chat UI |

---

## How a question flows through the system

1. You type a question in the browser (e.g. "Was plant die Stadt für die Neckarwiese?")
2. Next.js sends it to `/api/chat`
3. That route forwards it to `http://localhost:8000/chat`
4. `rag_server.py`:
   - Embeds your question with `multilingual-e5-base` (using the `"query: "` prefix, matching how your chunks were embedded with `"passage: "`)
   - Computes cosine similarity against all 1,307 chunk embeddings (dot product, since they're pre-normalized)
   - Takes the top 5 most relevant chunks
   - Builds a German-language prompt with that context
   - Sends it to Ollama's `/api/generate` endpoint
5. Ollama's local LLM generates an answer grounded in your STEK 2035 documents
6. The answer + source chunks are sent back and displayed, with sources collapsible under each reply

---

## Customization

**Change how many chunks are retrieved:**
In `rag_server.py`:
```python
TOP_K = 5   # increase for more context, decrease for faster/tighter answers
```

**Change the Ollama model:**
```python
OLLAMA_MODEL = "llama3.1"   # or "mistral", "qwen2.5:7b", etc.
```

**Adjust the system prompt (tone, language, strictness):**
Edit the `build_prompt()` function in `rag_server.py`.

---

## Troubleshooting

**"Verbindung zum Backend fehlgeschlagen"**
- Check terminal 2 — is `uvicorn` running without errors?
- Check `curl http://localhost:8000/health`

**Ollama connection refused**
- Run `ollama serve` manually if it's not auto-starting
- Verify the model is pulled: `ollama list`

**Embedding model download is slow / fails**
- It's a ~1GB one-time download from Hugging Face — needs internet access on first run only. After that it's cached locally and runs fully offline.

**Answers are generic / ignore the documents**
- Increase `TOP_K` to pull more context
- Check `chunks.jsonl` path — `meta.json` should report `chunks_loaded: 1307`

---

## Next steps

- Add streaming responses (Ollama supports `"stream": true`)
- Add a topic filter dropdown (your `chunks_v2.jsonl` already has `topics` tags)
- Persist chat history to a file or database
- Deploy the backend behind a reverse proxy if you want it accessible beyond localhost
