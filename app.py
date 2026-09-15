"""RAG Generator -- minimal runtime RAG app.

Upload documents, ask questions, get answers grounded in those documents.
Swap document sets at any time by clearing and re-ingesting -- no code
changes needed for a different set of documents.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from parser import extract_text
from chunker import chunk_text
from embedder import embed
from store import VectorStore
from generator import generate_answer

# Loads GEMINI_API_KEY (and anything else) from a local .env file if present.
# Harmless no-op if .env doesn't exist -- e.g. in production where the
# variable is set directly in the environment. Safe to call after the
# imports above since none of them read env vars at import time.
load_dotenv()

# Below this cosine similarity, a chunk is considered irrelevant and is
# excluded from context -- reinforces grounding at retrieval time, not
# just via prompt wording.
SIMILARITY_THRESHOLD = 0.3
TOP_K = 4

st.set_page_config(page_title="RAG Generator", layout="centered")
st.title("RAG Generator")

if not os.environ.get("GEMINI_API_KEY"):
    st.warning(
        "GEMINI_API_KEY is not set. You can still upload and ingest documents, "
        "but asking questions will fail until it's configured. "
        "Copy .env.example to .env and add your key, or set it in your shell. "
        "See README.md for details."
    )

if "store" not in st.session_state:
    st.session_state.store = VectorStore()
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# --- Ingestion ---
st.header("1. Ingest documents")
uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF, TXT, MD)",
    accept_multiple_files=True,
)

col1, col2 = st.columns(2)
with col1:
    ingest_clicked = st.button("Ingest", type="primary", disabled=not uploaded_files)
with col2:
    reset_clicked = st.button("Clear / start new document set")

if reset_clicked:
    st.session_state.store.clear()
    st.session_state.ingested_files = []
    st.success("Store cleared. Upload a new document set.")

if ingest_clicked:
    with st.spinner("Parsing, chunking, and embedding..."):
        errors = []
        for f in uploaded_files:
            try:
                text = extract_text(f.name, f.read())
                chunks = chunk_text(text)
                if not chunks:
                    errors.append(f"{f.name}: produced no chunks")
                    continue
                vectors = embed(chunks)
                metadata = [{"text": c, "source": f.name} for c in chunks]
                st.session_state.store.add(vectors, metadata)
                st.session_state.ingested_files.append(f.name)
            except ValueError as e:
                errors.append(f"{f.name}: {e}")

        if st.session_state.ingested_files:
            st.success(f"Ingested: {', '.join(st.session_state.ingested_files)}")
        for err in errors:
            st.error(err)

if st.session_state.ingested_files:
    st.caption(f"Current document set: {', '.join(st.session_state.ingested_files)}")

# --- Query ---
st.header("2. Ask a question")
question = st.text_input("Question")
ask_clicked = st.button("Ask", disabled=st.session_state.store.is_empty())

if st.session_state.store.is_empty():
    st.caption("Ingest a document set above before asking questions.")

if ask_clicked and question:
    with st.spinner("Retrieving and generating..."):
        query_vector = embed([question])[0]
        raw_results = st.session_state.store.search(query_vector, top_k=TOP_K)
        results = [r for r in raw_results if r["score"] >= SIMILARITY_THRESHOLD]

        try:
            answer = generate_answer(question, results)
        except RuntimeError as e:
            st.error(str(e))
            answer = None

        if answer is not None:
            st.subheader("Answer")
            st.write(answer)

            st.subheader("Retrieved sources")
            if results:
                for r in results:
                    st.markdown(f"**{r['source']}** (similarity: {r['score']:.3f})")
                    st.code(r["text"])
            else:
                st.caption(
                    f"No chunk cleared the similarity threshold "
                    f"({SIMILARITY_THRESHOLD}) -- nothing was passed to the model."
                )
