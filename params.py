# batch_size = 32
# block_size = 8
# max_iters = 5000
# n_embed = 32
# n_head = 4
# n_layer = 4
# dropout = 0.2
# learning_rate = 3e-4
batch_size = 64
block_size = 64
max_iters = 5000
n_embed = 256
n_head = 4
n_layer = 4
dropout = 0.2
learning_rate = 3e-4


input_file = open("./input.txt", "r", encoding="utf-8")
text = input_file.read()

# get unique chars
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s):
    return [stoi[ch] for ch in s]


def decode(l):
    return ''.join([itos[i] for i in l])
