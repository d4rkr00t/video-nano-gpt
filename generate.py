import torch
from model import BigramLanguageModel
from params import decode

model = BigramLanguageModel()
model.load_state_dict(torch.load('model.pt'))


context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generagte(context, max_new_tokens=500)[0].tolist()))
