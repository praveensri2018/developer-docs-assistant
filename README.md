# Developer Documentation Assistant

A personal proof of concept built with AI assistance, extending Praveen Bala's backend portfolio. LangGraph orchestrates retrieval, evidence gating, generation and abstention. FastAPI serves a small browser UI. Upload text PDFs, Markdown or TXT files and inspect source/page references.

## Quick start

Python 3.12 was used for verification.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Two fictional API/operations documents are loaded initially. Upload replaces the in-memory corpus; reset restores samples. Restart discards uploaded documents. No external calls occur in default extractive mode.

## How it works

1. Extract text, preserving PDF page numbers. Split into 220-word chunks with 40-word overlap.
2. Build local TF-IDF unigram/bigram vectors, normalize, and index with FAISS inner-product search.
3. LangGraph retrieves top 3 chunks and applies a configurable similarity threshold (default 0.12).
4. Low-evidence queries route to abstention. Other queries return exact excerpts by default or call the configured LLM.
5. Responses include source filenames, page numbers, scores and graph trace. Generated citations are checked for valid source IDs, not semantic correctness.

This version uses lexical TF-IDF vectors, **not pretrained semantic embeddings**. Synonyms and paraphrases can fail. It is a retrieval-grounded LLM application when an LLM provider is configured; default mode is an extractive retrieval demo.

## Optional generation

Set environment variables before starting the server (PowerShell syntax shown):

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="your-local-key"
$env:LLM_MODEL="gpt-4o-mini"
```

Or run a local Ollama instance with the selected model already pulled:

```powershell
$env:LLM_PROVIDER="ollama"
$env:OLLAMA_URL="http://localhost:11434"
$env:LLM_MODEL="llama3.2"
```

Never commit credentials. OpenAI mode sends your question and retrieved excerpts externally and can incur API charges. Ollama mode sends them to the configured Ollama host. Optional provider integrations are implemented but were **not live-tested** without user credentials or an available Ollama instance.

## Verification

```bash
python -m pytest -q
python evaluate.py
```

Five API/core tests cover retrieval, abstention, upload/reset, unsupported files, empty documents and missing-provider credentials. The included evaluation contains six answerable and two unrelated queries on two fictional sample documents: retrieval hit@3 6/6 and abstention 2/2 in extractive mode. These are smoke checks, not a generalization benchmark, LLM answer accuracy or citation-faithfulness measurement. Latency is local and excludes LLM calls.

## Limitations

Single-user local POC: no authentication, persistent index, tenant isolation, OCR or production deployment. All clients share one corpus; do not expose the server publicly. A lexical threshold cannot guarantee that a question is answerable, and a valid citation ID cannot guarantee a supported answer. Uploaded document text is treated as untrusted context in the generation prompt; this does not eliminate prompt injection. Model generation requires further live evaluation and human source checks. No model training, semantic embeddings, production usage or business impact is claimed.

## Files

- core.py: retrieval and LangGraph workflow
- app.py: API and document ingestion
- index.html: browser interface
- samples/: original fictional demonstration documents
- tests/: automated checks
- evaluate.py / evaluation.json: reproducible smoke evaluation
