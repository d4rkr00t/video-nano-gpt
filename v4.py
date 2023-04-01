import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

B, T, C = 4, 8, 32  # batch, time, channels
x = torch.randn(B, T, C)

# single head performing self attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)  # (B, T, head_size)
q = query(x)  # (B, T, head_size)
# (B, T, 16) @ (B, 16, T) -> (B, T, T)
wei = q @ k.transpose(-2, -1) * head_size**-0.5

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

v = value(x)
out = wei @ v

print(out.shape)
print(wei[0])
