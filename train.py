# https://www.youtube.com/watch?v=kCc8FmEb1nY

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 1e-3
device = 'cude' if torch.cuda.is_available() else 'cpu'
n_embed = 32
# ---------------

input_file = open("./input.txt", "r", encoding="utf-8")
text = input_file.read()

# get unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

#
# tokenize input text
#

# creating a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
# take a string output a list of integers


def encode(s):
    return [stoi[ch] for ch in s]
# tage a list of integers output a string


def decode(l):
    return ''.join([itos[i] for i in l])


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
    x, y = x.to(device), y.to(device)
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
# Self-attention
#


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.key(x)
        v = self.key(x)

        # compute attention scores
        wei = q @ k.transpose(-2, -1) * C**-0.5

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        return out

#
# Bigram language model
#


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.sa_head = Head(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensors of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.sa_head(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # (B*T,C)

            targets = targets.view(B * T)  # (B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generagte(self, idx, max_new_tokens):
        # idx is (B,T) array of indicies in the current target
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the prediction
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B,C)

            # apply softmax to get the probabilities
            propbs = F.softmax(logits, dim=-1)

            # sample from the distribution
            idx_next = torch.multinomial(propbs, num_samples=1)  # (B,1)

            # append sampled index th the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B,T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

#
# Optimise the model
#

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for step in range(max_iters):
    # every once and a while evaluate the loss on train and val sets
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#
# Generate from the model
#

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generagte(context, max_new_tokens=500)[0].tolist()))
