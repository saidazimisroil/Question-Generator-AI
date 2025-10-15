from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .gpt_api import DEFAULT_GPT_MODEL, generate_question_with_gpt

logger = logging.getLogger(__name__)

app = FastAPI(title="Uzbek Question Generation API", version="1.0.0")


class GenerateRequest(BaseModel):
    sentence: str = Field(..., description="Declarative Uzbek sentence to convert into a question.")
    model: str | None = Field(
        default=None,
        description="Optional override for the OpenAI chat model (defaults to gpt-3.5-turbo).",
    )


class GenerateResponse(BaseModel):
    sentence: str
    question: str
    model: str


@app.get("/")
def read_root() -> dict[str, str]:
    """Health check endpoint."""

    return {"status": "ok", "message": "Uzbek question generation service is running."}


@app.post("/generate", response_model=GenerateResponse)
def create_question(payload: GenerateRequest) -> GenerateResponse:
    """Generate a question from a sentence using the GPT API."""

    model = payload.model or DEFAULT_GPT_MODEL

    try:
        question = generate_question_with_gpt(payload.sentence, model=model)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Failed to generate question with GPT")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return GenerateResponse(sentence=payload.sentence, question=question, model=model)


__all__ = ["app"]
