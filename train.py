# https://www.youtube.com/watch?v=kCc8FmEb1nY

import torch
from torch.nn import functional as F
from model import BigramLanguageModel
from params import text, block_size, batch_size, max_iters, learning_rate, encode

torch.manual_seed(1337)

eval_iters = 200
eval_interval = 500

#
# tokenize input text
#


#
# tokenizing the entire input text dataset
#
data = torch.tensor(encode(text), dtype=torch.long)

#
# split train and validation sets
#

n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


#
#
# Helper functions
#
#


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()

    return out


#
# Bigram language model
#

model = BigramLanguageModel()

#
# Optimise the model
#

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("Training...")

for step in range(max_iters):
    # every once and a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(
    f"step {max_iters}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")


print("Saving model...")
torch.save(model.state_dict(), 'model.pt')

print("Done.")
