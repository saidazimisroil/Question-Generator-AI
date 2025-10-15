import torch
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('models/uz_qg_mt5')
base = AutoModelForSeq2SeqLM.from_pretrained('google/mt5-small')
total = 0
changed = 0
with torch.no_grad():
    for (n1,p1),(n2,p2) in zip(model.state_dict().items(), base.state_dict().items()):
        if p1.shape != p2.shape:
            continue
        total += 1
        if not torch.allclose(p1, p2):
            changed += 1
print('changed', changed, 'of', total)
