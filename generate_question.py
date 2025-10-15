from __future__ import annotations

import argparse
import logging

from src.question_generation.gpt_api import DEFAULT_GPT_MODEL, generate_question_with_gpt


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an Uzbek question using the GPT API.")
    parser.add_argument("--sentence", required=True, help="Declarative Uzbek sentence to turn into a question.")
    parser.add_argument(
        "--model",
        default=DEFAULT_GPT_MODEL,
        help="Optional OpenAI chat model identifier (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    try:
        question = generate_question_with_gpt(args.sentence, model=args.model)
    except Exception as exc:  # pragma: no cover - surface CLI error
        logger.error("Failed to generate question: %s", exc)
        raise SystemExit(1) from exc

    print(f"Matn: {args.sentence}")
    print(f"Savol: {question}")


if __name__ == "__main__":
    main()
