from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.question_generation.data import DatasetConfig, load_local_dataset
from src.question_generation.train import preprocess_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = DatasetConfig(data_path=Path('data/uz_qg_sample.jsonl'), validation_ratio=0.2, seed=42)
dataset = load_local_dataset(config)
tokenizer = AutoTokenizer.from_pretrained('models/uz_qg_mt5')
model = AutoModelForSeq2SeqLM.from_pretrained('models/uz_qg_mt5').to(device)
model.eval()
for row in dataset['train']:
    sentence = row['input_text']
    target = row['target_text']
    inputs = tokenizer(sentence, return_tensors='pt').to(device)
    output_ids = model.generate(**inputs, max_length=64, num_beams=4, do_sample=False)
    pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print('INPUT:', sentence)
    print('TARGET:', target)
    print('PRED:', pred)
    print('-'*40)
