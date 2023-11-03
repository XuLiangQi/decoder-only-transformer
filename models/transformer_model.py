import torch
import torch.nn as nn
from torch.nn import functional as F

from tools.get_batch import get_batch

batch_size = 4  # How many independent sequences will we precess in parallel
block_size = 8  # What is the maximum context length for predictions
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
vocab_size = 65
n_embd = 32

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=1)

        out = wei @ v

        return out

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, n_embd),
                                    nn.ReLU())
    
    def forward(self, x):
        return self.net(x)

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        # Initializing the embedding table with size of vocab_size**2
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.head = MultiHead(4, int(n_embd/4))
        self.net = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)    # (B, T, C), in this case (4 (batch_size), 8 (block_size), 65(vocab_size))
                                                     # becase idx is (B, T), and each element from "idx" will go in token_embedding_table
                                                     # and get a row of data (1, 65(C))
        pos_emb = self.position_embedding_table(torch.arange(T))       # (T, C)
        x = tok_emb + pos_emb       # (B, T, C)
        x = self.head(x)
        x = self.net(x)
        logits = self.lm_head(x)               # (B, T, vocab_size)

        if targets is not None:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)        # Returns a new tensor with the same data as the self tensor but of a different shape.
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, _ = self.forward(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :]   # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1)     # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1)    # (B, 1)
            # Append sampled index to the running sequencing
            idx = torch.cat((idx, idx_next), dim = 1)      # (B, T+1)

        return idx

    def train(self, train_data, batch_size, block_size):
        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate)
        # batch_size = 32
        for iter in range(max_iters):
            # Sample a batch of data
            xb, yb = get_batch(train_data, batch_size, block_size)

            # Evaluate the loss
            _, loss = self.forward(xb, yb)
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

            # Print loss every 1000 steps
            if iter % 1000 == 0:
                print(loss.item())