"""Generate an answer grounded strictly in retrieved context chunks.

Two responsibilities kept separate on purpose:
  - build_user_message: pure prompt construction, no network call, fully
    testable without an API key.
  - generate_answer: the actual LLM call (Gemini).

Grounding is enforced structurally, not just by prompt wording: if there
are no context chunks (e.g. filtered out by a similarity threshold
upstream), the LLM is never called at all.
"""

import os

from google import genai
from google.genai import types

MODEL_NAME = "gemini-3.6-flash"  # swap here if you have access to a newer one
NO_ANSWER_MESSAGE = "I couldn't find relevant information in the provided documents."

SYSTEM_PROMPT = (
    "You are a question-answering assistant. Answer strictly and only using "
    "the provided context -- never use outside knowledge, even if you know "
    "the answer. If the context does not contain enough information to "
    f'answer, respond with exactly: "{NO_ANSWER_MESSAGE}" '
    "Keep answers concise."
)

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Set it before asking questions."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def build_user_message(question: str, context_chunks: list[dict]) -> str:
    """Construct the user-turn prompt from retrieved chunks. No network call."""
    context_block = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    return (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """Return an answer grounded in context_chunks, or the no-answer message.

    If context_chunks is empty, the LLM is never called -- grounding failure
    is handled structurally rather than relying on the model to refuse.
    """
    if not context_chunks:
        return NO_ANSWER_MESSAGE

    client = _get_client()
    user_message = build_user_message(question, context_chunks)
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            max_output_tokens=500,
        ),
    )
    return response.text
