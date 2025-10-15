# Uzbek Question Generation | GPT API based type

This project provides a lightweight workflow for training and serving a sequence-to-sequence model that turns Uzbek statements into questions. The default configuration fine-tunes `google/mt5-small` on a small handcrafted dataset, but you can extend it with your own high-quality sentence/question pairs.

## Project layout

- `data/uz_qg_sample.jsonl` &mdash; starter dataset with Uzbek examples (`input_text`, `target_text`).
- `src/question_generation/data.py` &mdash; helpers that load and split local JSONL files into Hugging Face datasets.
- `src/question_generation/train.py` &mdash; fine-tuning script built on `transformers.Seq2SeqTrainer`.
- `src/question_generation/inference.py` &mdash; CLI for generating questions with a trained checkpoint.
- `requirements.txt` &mdash; main Python dependencies.

## Setup

Create a virtual environment (recommended) and install requirements:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

> **Note:** Install the correct CUDA-enabled `torch` wheel if you plan to train on GPU.

## Training

Fine-tune the model on the provided dataset:

```bash
python -m src.question_generation.train \
  --train_file data/uz_qg_sample.jsonl \
  --model_name_or_path google/mt5-small \
  --output_dir models/uz_qg_mt5 \
  --source_prefix "uz_qg: " \
  --num_train_epochs 50 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 1e-4 \
  --validation_ratio 0.2 \
  --label_smoothing_factor 0.1 \
  --weight_decay 0.01 \
  --max_grad_norm 1.0
```

The script now applies consistent task prompting (`uz_qg: `), safer generation defaults (beam search, no-repeat n-grams), and optional label smoothing. Adjust the hyperparameters as you scale up the dataset; on tiny corpora, more epochs with a low learning rate usually work best.

## Inference

After training, generate questions with:

```bash
python -m src.question_generation.inference \
  --model_path models/uz_qg_mt5 \
  --source_prefix "uz_qg: " \
  --sentences "Bu Toshkent O'zbekistonning poytaxti." \
  --sentences "Talaba kutubxonada imtihonga tayyorlanmoqda."
```

Sample output:

```
Matn: Bu Toshkent O'zbekistonning poytaxti.
Savol: O'zbekistonning poytaxti qaysi shahardir?

Matn: Talaba kutubxonada imtihonga tayyorlanmoqda.
Savol: Talaba qayerda imtihonga tayyorlanmoqda?
```

You can also provide an input file (one sentence per line):

```bash
python -m src.question_generation.inference \
  --model_path models/uz_qg_mt5 \
  --input_file sentences.txt
```

By default inference runs with deterministic beam search (`--no_sample`) plus post-processing that strips `<extra_id_*>` markers, collapses repeated punctuation, and guarantees the result ends with a question mark. Add `--sample` if you want more diverse questions.

## Extending the dataset

The model quality depends on the diversity and correctness of the training pairs. To improve it:

1. Curate sentence/question pairs from Uzbek educational materials or crowdsourced annotations.
2. Translate existing high-quality question generation datasets (e.g., SQuAD) into Uzbek, then manually clean.
3. Include metadata such as answer spans if you plan to generate answer-aware questions; extend `data.py` accordingly.
4. Re-run the training script with the updated JSONL file.

## Tips

- Start with `google/mt5-base` or `facebook/mbart-large-50` once you have enough data; larger models usually yield better grammar.
- Monitor TensorBoard logs in `models/uz_qg_mt5/runs` to spot overfitting.
- Experiment with generation parameters (`--num_beams`, `--temperature`, `--top_p`) for different question styles.
