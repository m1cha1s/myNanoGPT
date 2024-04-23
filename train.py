#!/usr/bin/python3

from sys import stdout
from datetime import datetime
import logging
import argparse

logging.basicConfig(filename=f'trainingLogs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log', encoding='utf-8', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stdout))

parser = argparse.ArgumentParser(description='Trains a neural network.')

parser.add_argument('--model')
parser.add_argument('--training-data')
parser.add_argument('--iters')
parser.add_argument('--learning-rate')

args = parser.parse_args()

import torch
import torch.nn as nn
from torch.nn import functional as F

# torch.manual_seed(1337) # For reproducability

batch_size = 64
block_size = 256
max_iters = 5000 if args.iters == None else int(args.iters)
eval_interval = 500
learning_rate = 3e-4 if args.learning_rate is None else float(args.learning_rate)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2


with open('input.txt' if args.training_data == None else args.training_data, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

# Splitting the dataset into train and validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Make sure that the split doesn't overlap
assert(len(train_data)+len(val_data)==len(data))

batch_size = 4 # how many independed sequences will we process in parallel
block_size = 8 # what is the maximum context length for predictions?


def get_batch(*, train=False):
    # generate a small batch of data of inputs x and targets y
    data = train_data if train else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y


@torch.no_grad
def estimate_loss(model):
    out = {}
    model.eval()
    for split in [True, False]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train=split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out['train' if split else 'val'] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ A single head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, head_count, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)])
        self.proj = nn.Linear(n_embed, n_embed)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head) -> None:
        super().__init__()

        head_size = n_embed//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embed),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        position_embedding = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeddings + position_embedding
        x = self.blocks(x)
        logits = self.lm_head(x) # (B,T,vocab_size)


        if targets is None:
            loss = None
        else:
            # We need to reshape logits and targets so that cross_entropy can take them
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # Crop idx to last block_size token
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if args.model == None:
    logging.debug("Training from scratch...")
    m = BigramLanguageModel().to(device)
else:
    logging.debug("Continuing training...")
    m = torch.load(args.model)


# ---- training ----

logging.info(f"The model will run on {device}")

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
m.train() # Set model to train mode

from time import perf_counter

it_time = 0

for iter in range(max_iters):

    start_time = perf_counter()

    if iter % eval_interval == 0:
        losses = estimate_loss(m)
        logging.info(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, iteration time {it_time:.4f}")

    xb, yb = get_batch(train=True)

    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if it_time==0:
        it_time = perf_counter()-start_time
    else:
        it_time += perf_counter()-start_time
        it_time /= 2

m.eval() # Set model to eval mode

torch.save(m, f"model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.model")

idx = torch.zeros((1,1), dtype=torch.long, device=device)
output = decode(m.generate(idx,max_new_tokens=10000)[0].tolist())

with open("output.txt", "w") as o:
    o.write(str(output))