from __future__ import annotations

import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Load environment variables from a local .env file (if present) before we read them.
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_GPT_MODEL = os.getenv("QUESTION_GPT_MODEL", "gpt-3.5-turbo")
SYSTEM_PROMPT = (
    "You are an assistant that generates well-formed Uzbek questions based on "
    "a provided declarative sentence."
)
USER_PROMPT_TEMPLATE = (
    "Convert the following Uzbek statement into a single clear question in Uzbek. "
    "Respond with the question only.\n\nMatn: {sentence}"
)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    """Create a cached OpenAI client using credentials from the environment."""

    return OpenAI()


def build_messages(sentence: str) -> list[dict[str, str]]:
    """Construct chat messages for the GPT request."""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(sentence=sentence)},
    ]


def generate_question_with_gpt(sentence: str, model: str = DEFAULT_GPT_MODEL) -> str:
    """Call the GPT API to transform a sentence into a question."""

    if not sentence.strip():
        raise ValueError("Sentence must not be empty.")

    client = _get_client()

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=build_messages(sentence),
            temperature=0.7,
            max_tokens=128,
        )
    except OpenAIError as exc:  # pragma: no cover - network call error propagation
        logger.exception("OpenAI API request failed")
        raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

    choice = completion.choices[0]
    message = choice.message
    if message is None or message.content is None:
        raise RuntimeError("OpenAI API returned an empty response.")

    question = message.content.strip()
    if not question.endswith("?"):
        question = question.rstrip(". ") + "?"
    return question


__all__ = ["generate_question_with_gpt", "DEFAULT_GPT_MODEL"]
