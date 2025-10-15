from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


logger = logging.getLogger(__name__)
DEFAULT_PREFIX = "uz_qg: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Uzbek questions from statements.")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=Path("models/uz_qg_mt5"),
        help="Path to a fine-tuned model directory or a public checkpoint.",
    )
    parser.add_argument("--sentences", type=str, nargs="*", help="One or more sentences to convert into questions.")
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="Prefix used during fine-tuning (should match training).",
    )
    parser.add_argument(
        "--input_file",
        type=Path,
        help="Optional text file containing one sentence per line.",
    )
    parser.add_argument("--max_length", type=int, default=0, help="Optional absolute max length for outputs (0 = auto).")
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_new_tokens", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=48)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--length_penalty", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--sample", action="store_true", help="Enable sampling for diverse generations.")
    parser.add_argument("--no_sample", action="store_true", help="Disable sampling and use deterministic decoding.")
    parser.add_argument("--no_early_stopping", action="store_true", help="Disable beam-search early stopping.")
    parser.set_defaults(no_sample=True)
    args = parser.parse_args()
    if args.sample:
        args.no_sample = False
    args.early_stopping = not args.no_early_stopping
    return args


def load_sentences(args: argparse.Namespace) -> Iterable[str]:
    sentences = list(args.sentences or [])

    if args.input_file:
        text = args.input_file.read_text(encoding="utf-8").splitlines()
        sentences.extend(line.strip() for line in text if line.strip())

    if not sentences:
        raise ValueError("No sentences provided. Pass via --sentences or --input_file.")

    return sentences


def generate(
    model_path: Path | str,
    sentences: Iterable[str],
    max_length: int,
    num_beams: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    min_new_tokens: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int,
    length_penalty: float,
    repetition_penalty: float,
    early_stopping: bool,
    source_prefix: str,
) -> Iterable[tuple[str, str]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id

    for sentence in sentences:
        prefixed = f"{source_prefix}{sentence}" if source_prefix else sentence
        inputs = tokenizer(prefixed, return_tensors="pt", padding=False).to(device)
        sequence_length = inputs["input_ids"].shape[-1]
        max_length_value = max_length if max_length > 0 else sequence_length + max_new_tokens
        min_length_value = max(sequence_length + min_new_tokens, 1)
        if min_length_value > max_length_value:
            max_length_value = min_length_value
        generation_kwargs = {
            "max_length": max_length_value,
            "min_length": min_length_value,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "early_stopping": early_stopping,
        }
        if do_sample:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})
        else:
            generation_kwargs["do_sample"] = False
        with torch.no_grad():
            output_ids = model.generate(**inputs, **generation_kwargs)
        question = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        yield sentence, post_process_question(question)


def post_process_question(text: str) -> str:
    question = text.strip()
    if not question:
        return question
    # Remove sentinel tokens like <extra_id_0>
    question = re.sub(r"<extra_id_\d+>", "", question)
    # Normalize whitespace and punctuation spacing
    question = re.sub(r"\s+", " ", question)
    question = question.replace(" ?", "?").replace(" ,", ",").strip()
    # Collapse multiple question marks
    question = re.sub(r"\?{2,}", "?", question)
    # Ensure the string ends with a question mark
    if not question.endswith("?"):
        question = question.rstrip(". ")
        question = question + "?"
    # Capitalize first letter if needed
    if question:
        question = question[0].upper() + question[1:]
    return question


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sentences = load_sentences(args)
    do_sample = not args.no_sample

    for sentence, question in generate(
        model_path=args.model_path,
        sentences=sentences,
        max_length=args.max_length,
        num_beams=args.num_beams,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=do_sample,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        early_stopping=args.early_stopping,
        source_prefix=args.source_prefix,
    ):
        logger.info("Matn: %s", sentence)
        logger.info("Savol: %s\n", question)


if __name__ == "__main__":
    main()
