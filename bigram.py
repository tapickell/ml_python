"""
Super simple BiGram model for educational purposes
Decode only transformer
"""
import torch
from torch import nn
from torch.nn import functional as F
import pprint

# https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# hyperparamters
BATCH_SIZE = 64 # 32 # 64
BLOCK_SIZE = 256 #8  # 256
MAX_ITERS = 5000 # increase with lowered learning rate
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-4 # self-attention cannot tolerate high learning rates
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 200
N_EMBED = 384 # 32 # 384 ???
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2

class Head(nn.Module):
    "One head of self-attention"

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        key = self.key(x).transpose(-2, -1) # (B, C, T)
        query = self.query(x) # (B, T, C)
        normalization = C**-0.5
        w = query @ key # (B, T, C) @ (B, C, T) -> (B, T, T)
        w *= normalization # (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights =  F.softmax(w, dim=-1) # (B, T, T)
        out = weights @ self.value(x) # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    "Multiple self-attention heads in parallel"

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(head_size * num_heads, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        catted = torch.cat([h(x) for h in self.heads], dim=-1) # -1 is channel dimmension
        out = self.dropout(self.projection(catted))
        return out

class FeedForward(nn.Module):
    "Simple linear layer followed by non-linearity"

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    "Transformer block; communication followed by computation"

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    "Super simple BiGram Model"

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED) # Final layer normalization
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.2)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embed = self.token_embedding_table(idx)  # (B,T,C)
        pos_embed = self.position_embedding_table(
            torch.arange(T, device=DEVICE)
        )  # (T,C)
        embeddings = tok_embed + pos_embed
        embeddings = self.blocks(embeddings) # apply one head of self-attention (B, T, C)
        embeddings = self.ln_f(embeddings)
        logits = self.lm_head(embeddings)  # apply decoder lm head (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        "generate/2"
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            blocked_idx = idx[:, -BLOCK_SIZE:] # crops idx context to be block size
            logits, loss = self(blocked_idx)
            logits = logits[:, -1, :]  # => (B, C)
            probs = F.softmax(logits, dim=-1)  # => (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # => (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


def get_batch(data):
    "get_batch/1 splits data to batches, and puts data on device"
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    "loss estimation function using mean"
    out = {}
    model.eval()
    for (key, data) in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[key] = losses.mean()
    model.train()
    return out


def encode(work, mapping):
    "returns encode/decode from mapping"
    return [mapping[i] for i in work]


def get_text():
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return text


# RUN IT
text = get_text()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Simple Index Based Encoding Lookup Tables
CHAR_TO_INT = { ch:i for i,ch in enumerate(chars) }
INT_TO_CHAR = { i:ch for i,ch in enumerate(chars) }

data = torch.tensor(encode(text, CHAR_TO_INT), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

model = BigramLanguageModel(vocab_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

for iteration in range(MAX_ITERS):
    if iteration % EVAL_INTERVAL == 0:
        losses = estimate_loss(model, train_data, val_data)
        print(f"step {iteration}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    x_sample, y_sample = get_batch(train_data)
    logits, loss = model(x_sample, y_sample)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
pprint.pprint(encode(model.generate(idx, max_new_tokens=500)[0].tolist(), INT_TO_CHAR))


