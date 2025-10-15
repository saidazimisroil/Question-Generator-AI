from pathlib import Path
from transformers import AutoTokenizer
from src.question_generation.data import DatasetConfig, load_local_dataset
from src.question_generation.train import preprocess_dataset

config = DatasetConfig(data_path=Path('data/uz_qg_sample.jsonl'), validation_ratio=0.2, seed=42)
dataset = load_local_dataset(config)
tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
processed = preprocess_dataset(dataset, tokenizer, config.input_field, config.target_field, 128, 64)
print(processed)
print(processed['train'][0])
print(tokenizer.decode(processed['train'][0]['input_ids'], skip_special_tokens=True))
labels = [token for token in processed['train'][0]['labels'] if token != -100]
print(tokenizer.decode(labels, skip_special_tokens=True))
