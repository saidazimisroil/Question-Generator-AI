#!/usr/bin/env python3
"""Generate questions from text using a fine-tuned question generation model.

Usage examples:
  python predict.py "Toshkent O'zbekistonning poytaxti hisoblanadi."
  python predict.py --model_path models/uz_qg_mt5_v7 "Amir Temur 1336-yilda tug'ilgan."
  echo "Alisher Navoiy buyuk shoir." | python predict.py

Options:
  --model_path   Path to fine-tuned model directory (default: models/uz_qg_mt5_v7)
  --num_beams    Beam size for generation (default: 4)
"""
import argparse
import sys
from pathlib import Path

from src.question_generation.inference import generate, post_process_question


def main():
    p = argparse.ArgumentParser(description="Generate questions from text using a question generation model")
    p.add_argument("text", nargs="*", help="Text/sentence to generate a question from (or omit to read from stdin)")
    p.add_argument("--model_path", type=Path, default=Path("models/uz_qg_mt5_v7"), 
                   help="Path to fine-tuned model directory (default: models/uz_qg_mt5_v7)")
    p.add_argument("--source_prefix", type=str, default="uz_qg: ", 
                   help="Prefix used during fine-tuning (should match training)")
    p.add_argument("--num_beams", type=int, default=4, help="Beam size for generation")
    p.add_argument("--max_new_tokens", type=int, default=48, help="Maximum newly generated tokens")
    p.add_argument("--min_new_tokens", type=int, default=1, help="Minimum newly generated tokens")
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    p.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling")
    p.add_argument("--length_penalty", type=float, default=0.9, help="Length penalty for beam search")
    p.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    p.add_argument("--no_repeat_ngram_size", type=int, default=3, help="N-gram size to prevent repetition")
    p.add_argument("--sample", action="store_true", help="Enable sampling for diverse generations")
    args = p.parse_args()

    # Get input text
    text = " ".join(args.text).strip()
    if not text:
        if sys.stdin and not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            try:
                text = input("Text: ").strip()
            except EOFError:
                text = ""

    if not text:
        print("No text provided.", file=sys.stderr)
        return 2

    try:
        # Generate question using the question generation model
        for sentence, question in generate(
            model_path=args.model_path,
            sentences=[text],
            max_length=0,
            num_beams=args.num_beams,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.sample,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens=args.max_new_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            length_penalty=args.length_penalty,
            repetition_penalty=args.repetition_penalty,
            early_stopping=True,
            source_prefix=args.source_prefix,
        ):
            print(f"Matn: {sentence}")
            print(f"Savol: {question}")
            
    except Exception as e:
        print(f"Generation error: {e}", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
