# Linux setup

Tested on Debian 13 (Python 3.13). Any glibc Linux with Python 3.11+ works; the steps below
use `python3` and `pip` from a virtualenv.

## 1. Install Python and a venv

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
python3 --version          # >= 3.11
```

Other distros: `dnf install python3 python3-pip`, `pacman -S python`, or `pyenv install 3.12`.

## 2. Get the code and install

```bash
git clone <your-fork-url> engineering-rag-assistant
cd engineering-rag-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"        # pytest, coverage, playwright (screenshots)
pip install -e ".[semantic]"   # sentence-transformers (downloads a ~90 MB model on first use)
pip install -e ".[ocr]"        # pytesseract wrapper; also needs the binary below
sudo apt-get install -y tesseract-ocr        # only for scanned PDFs
```

`tesseract-ocr` is genuinely optional: without it, text-less pages are reported as
`needs_ocr` and stay non-searchable rather than silently appearing empty.

## 3. Try it with the synthetic corpus

```bash
engrag build-demo           # writes data/demo_generated/*.pdf and indexes them
engrag ask "What are the rated voltage and frequency of DEMO-TR-01?"
```

You should see a numbered extractive answer with `[D… p…:q…]` markers, then per-value
verification lines and a support verdict.

## 4. Run the app

```bash
engrag api &                # http://127.0.0.1:8502   (optional)
engrag web                  # http://127.0.0.1:8501
```

Both bind `0.0.0.0` by default. On a shared machine, restrict it:
`engrag web --host 127.0.0.1`, and for the API set `ENGRAG_HOST=127.0.0.1`.

## 5. Systemd unit (optional, keeps it running)

`/etc/systemd/system/engrag-web.service`:

```ini
[Unit]
Description=Engineering RAG assistant (Streamlit)
After=network.target

[Service]
User=engrag
WorkingDirectory=/opt/engineering-rag-assistant
Environment=ENGRAG_DATA_DIR=/var/lib/engrag
ExecStart=/opt/engineering-rag-assistant/.venv/bin/python -m engrag web
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo mkdir -p /var/lib/engrag && sudo chown engrag: /var/lib/engrag
sudo systemctl daemon-reload && sudo systemctl enable --now engrag-web
```

Add a second unit for `-m engrag api` if you want the HTTP API. Keep both behind a reverse
proxy with TLS and authentication before exposing them beyond localhost — the application has
no accounts and no per-user document ACLs.

## 6. Verify the installation

```bash
python -m pytest tests -q                                  # 86 passed
python scripts/run_eval.py --json-out /tmp/eval.json       # 18/18
```

The browser checks need Chromium once: `python -m playwright install --with-deps chromium`,
then `python scripts/ui_e2e.py --url http://127.0.0.1:8501` while the app is running.

## 7. Docker instead of a venv (optional)

```bash
docker compose up --build      # UI → http://localhost:8501, API → http://localhost:8502/docs
```

The image builds from the repository and keeps all state in the `engrag-data` volume, so
`docker compose down` never loses your documents.

## 8. Bring your own documents

```bash
curl -F "file=@/path/to/manual.pdf" http://127.0.0.1:8502/api/v1/documents
curl -X POST http://127.0.0.1:8502/api/v1/ask \
     -H 'content-type: application/json' \
     -d '{"question":"What is the short-time withstand current rating?","top_k":8}'
```

or drag the PDF into **🗄 Document Library**. Duplicate uploads are refused with the id of the
copy already stored; deleting a document removes its file, chunks, FTS rows, vectors and render
cache, and marks any stored citations that pointed at it as stale.

## Troubleshooting (Linux)

| Symptom | Cause and fix |
|---|---|
| `sqlite3.OperationalError: no such module: fts5` | Distro Python built without FTS5. The app falls back to `LIKE` search automatically (`/health` shows `"fts5": false`); install a Python with FTS5 (pyenv, uv, deadsnakes) for real BM25. |
| `Address already in use` on 8501/8502 | `engrag web --port 8601` / `ENGRAG_API_PORT=8602`. |
| Frontend can't reach the API in a browser | Intentional: the UI calls the service in-process, so only the Streamlit port is required. |
| Semantic search does nothing | `sentence-transformers` not installed → the hashing n-gram backend is used instead (reported in `/health` as `vectors.backend: hashing-ngrams`). |
| Scanned PDF returns no text | Page flagged `needs_ocr`; install tesseract, enable OCR in AI Settings, then reindex. |
| Slow first answer | The DB is opened and embeddings computed lazily; later queries on the same process are much faster (median ≈ 0.1 s in this repo's measurement). |
