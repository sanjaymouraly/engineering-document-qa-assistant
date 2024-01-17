# Engineering Document Q&A Assistant

Answers questions over engineering documents with citations and verifiable source passages.

## Features

- Answers questions over engineering documents with an offline-first pipeline.
- Combines hybrid search, extractive answering, and citation verification.
- Checks quotes byte-for-byte against the cited chunk and physical page.
- Understands engineering units, tolerances, rated versus maximum values, and model variants.
- Keeps LLM providers optional and off until explicitly configured.
- Exposes a CLI, HTTP API, and Streamlit interface over the same service layer.

## Stack

Python, FastAPI, Uvicorn, Pydantic, HTTPX, Streamlit, pypdf, pdfplumber, Pillow, NumPy

## Clone

```bash
git clone https://github.com/sanjaymouraly/engineering-document-qa-assistant.git
cd engineering-document-qa-assistant
```

## Setup

```bash
# The project ships a Docker Compose setup.
docker compose up --build
```

See `docs/` or the compose file for environment variables and ports.

## Run

```bash
docker compose up --build
```

## Tests

```bash
pytest -q
```

## Notes

- Sample data, when included, is synthetic.

## License

MIT © 2024 [Sanjay Mouraly](https://github.com/sanjaymouraly)
