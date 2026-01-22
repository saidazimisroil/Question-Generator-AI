from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import evaluate
from datasets import DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .data import DatasetConfig, load_local_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune mT5 for Uzbek question generation.")
    parser.add_argument(
        "--train_file",
        type=Path,
        default=Path("data/uz_qg_sample.jsonl"),
        help=(
            "Path to a JSONL dataset file. Directories or glob patterns can be provided to "
            "load multiple files."
        ),
    )
    parser.add_argument(
        "--train_files",
        type=Path,
        nargs="+",
        default=(),
        help=(
            "Additional dataset files, directories, or glob patterns to combine with --train_file."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="google/mt5-small",
        help="Pretrained model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("models/uz_qg_mt5"),
        help="Directory where checkpoints and logs will be saved.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.2,
        help="Share of examples reserved for validation.",
    )
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument(
        "--source_prefix",
        type=str,
        default="uz_qg: ",
        help="Optional text prepended to every input statement (helps task conditioning).",
    )
    parser.add_argument(
        "--label_smoothing_factor",
        type=float,
        default=0.0,
        help="Amount of label smoothing to apply during training.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping value applied after each optimizer step.",
    )
    parser.add_argument(
        "--generation_num_beams",
        type=int,
        default=4,
        help="Number of beams used for generation during evaluation.",
    )
    parser.add_argument(
        "--generation_min_new_tokens",
        type=int,
        default=8,
        help="Minimum number of new tokens to generate when evaluating.",
    )
    parser.add_argument(
        "--generation_max_new_tokens",
        type=int,
        default=48,
        help="Maximum number of new tokens to generate when evaluating.",
    )
    parser.add_argument(
        "--generation_no_repeat_ngram_size",
        type=int,
        default=3,
        help="Prevent repeating n-grams of this size when generating.",
    )
    parser.add_argument(
        "--generation_length_penalty",
        type=float,
        default=0.9,
        help="Length penalty applied to beam scores during generation.",
    )
    parser.add_argument(
        "--generation_repetition_penalty",
        type=float,
        default=1.1,
        help="Penalize repeated tokens during generation.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training where supported.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the fine-tuned model to the Hugging Face Hub (requires login).",
    )
    return parser.parse_args()


def preprocess_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    input_field: str,
    target_field: str,
    max_source_length: int,
    max_target_length: int,
    source_prefix: str = "",
) -> DatasetDict:
    prefix = source_prefix or ""

    def preprocess_function(batch: Dict[str, Any]) -> Dict[str, Any]:
        inputs = [prefix + text for text in batch[input_field]]
        model_inputs = tokenizer(
            inputs,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[target_field],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    processed = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    return processed


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")
    raw_data_paths = [args.train_file, *args.train_files]
    logger.info("Loading dataset from %s", ", ".join(str(path) for path in raw_data_paths))

    dataset_config = DatasetConfig(
        data_paths=tuple(raw_data_paths),
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    dataset = load_local_dataset(dataset_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id
    model.config.num_beams = args.generation_num_beams
    model.config.no_repeat_ngram_size = args.generation_no_repeat_ngram_size
    model.config.length_penalty = args.generation_length_penalty
    model.config.repetition_penalty = args.generation_repetition_penalty
    model.config.min_new_tokens = args.generation_min_new_tokens
    model.config.max_new_tokens = args.generation_max_new_tokens

    tokenized_dataset = preprocess_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        input_field=dataset_config.input_field,
        target_field=dataset_config.target_field,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        source_prefix=args.source_prefix,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        metric_result = bleu.compute(predictions=decoded_preds, references=decoded_labels)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        return {
            "sacrebleu": metric_result["score"],
            "gen_len": float(np.mean(prediction_lens)),
        }

    def build_training_args_kwargs() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "output_dir": str(args.output_dir),
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "predict_with_generate": True,
            "num_train_epochs": args.num_train_epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "report_to": ["tensorboard"],
            "fp16": args.fp16,
            "seed": args.seed,
            "push_to_hub": args.push_to_hub,
            "generation_max_length": args.max_target_length,
            "label_smoothing_factor": args.label_smoothing_factor,
            "max_grad_norm": args.max_grad_norm,
            "generation_num_beams": args.generation_num_beams,
            "generation_min_new_tokens": args.generation_min_new_tokens,
            "generation_max_new_tokens": args.generation_max_new_tokens,
            "generation_length_penalty": args.generation_length_penalty,
        }

        if args.eval_steps > 0:
            kwargs["evaluation_strategy"] = "steps"
            kwargs["save_strategy"] = "steps"
            kwargs["load_best_model_at_end"] = True
            kwargs["metric_for_best_model"] = "sacrebleu"
            kwargs["greater_is_better"] = True
        else:
            kwargs["evaluation_strategy"] = "no"
            kwargs["save_strategy"] = "no"
            kwargs["load_best_model_at_end"] = False

        return kwargs

    def filter_supported_kwargs(
        cls: type, kwargs: Dict[str, Any], drop_callback: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        signature = inspect.signature(cls.__init__)
        parameters = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            return kwargs

        allowed = {name for name in parameters.keys() if name != "self"}
        filtered: Dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in allowed:
                filtered[key] = value
            else:
                if drop_callback:
                    drop_callback(key)
        return filtered

    def log_dropped_argument(arg_name: str) -> None:
        logger.warning("Dropping unsupported training argument: %s", arg_name)

    training_args_kwargs = build_training_args_kwargs()
    training_args_kwargs = filter_supported_kwargs(Seq2SeqTrainingArguments, training_args_kwargs, log_dropped_argument)

    # Align strategies when older transformers versions drop some arguments.
    if "evaluation_strategy" not in training_args_kwargs:
        if training_args_kwargs.get("save_strategy") == "steps":
            logger.warning("Falling back to `save_strategy='no'` because evaluation strategy is unsupported.")
            training_args_kwargs["save_strategy"] = "no"
        if training_args_kwargs.pop("load_best_model_at_end", False):
            logger.warning("Disabling `load_best_model_at_end` because evaluation strategy is unsupported.")
            training_args_kwargs.pop("metric_for_best_model", None)
            training_args_kwargs.pop("greater_is_better", None)
    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    if args.eval_steps > 0 and getattr(trainer, "eval_dataset", None) is not None:
        logger.info("Running final evaluation...")
        final_metrics = trainer.evaluate()
        for key, value in final_metrics.items():
            logger.info("Eval %s: %s", key, value)

    logger.info("Saving final model to %s", args.output_dir)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
