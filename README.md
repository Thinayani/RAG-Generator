# RAG Generator

Upload documents, ask questions, get answers grounded in those documents.
Works with any document set at runtime -- no code changes needed to switch
what you're asking questions about.

## Setup

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Configure your API key:

   ```
   cp .env.example .env
   ```

   Then open `.env` and replace the placeholder with your real key:

   ```
   GEMINI_API_KEY=your-actual-key-here
   ```

   Get a key at https://aistudio.google.com/apikey. `.env` is gitignored --
   it will not be committed.

3. Run the app:

   ```
   streamlit run app.py
   ```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Used to call the Gemini API for grounded answer generation. Without it, you can still upload and ingest documents, but asking questions will fail with a clear error telling you to set this. |

No other configuration is required. The embedding model (`sentence-transformers`,
`all-MiniLM-L6-v2`) runs locally and needs no API key -- it downloads its
weights once from Hugging Face on first use.

## Configuring the key in other environments

- **Local development**: use the `.env` file as above (loaded automatically
  via `python-dotenv`).
- **Deployed / hosted (e.g. Streamlit Community Cloud)**: set `GEMINI_API_KEY`
  as a secret/environment variable in the platform's settings rather than
  shipping a `.env` file. Streamlit Cloud specifically uses
  `.streamlit/secrets.toml`, which is also gitignored here.
- **Shell session**: `export GEMINI_API_KEY=your-key` before running the app.

In every case, the app reads the key from the environment at the moment a
question is asked -- it is never hardcoded, logged, or stored anywhere in
the codebase.

## What happens if the key is missing

The app checks for `GEMINI_API_KEY` on startup and shows a warning banner if
it's absent, so you know before you start uploading documents. Document
ingestion still works without it. If you try to ask a question without the
key configured, you'll get a clear in-app error rather than a crash or a
confusing third-party API error.
