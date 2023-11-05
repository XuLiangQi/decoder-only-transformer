import torch
import torch.nn as nn
from torch.nn import functional as F

from tools.get_batch import get_batch

batch_size = 4              # How many independent sequences will we precess in parallel
block_size = 8              # What is the maximum context length for predictions
max_iters = 5000            # How many steps the model will be training for
eval_interval = 300         # Print out the loss every "eval_interval" steps
learning_rate = 1e-3        # The model's learning rate
eval_iters = 200
vocab_size = 65             # Length of the vector of each token after embedding
embedding_dimensions = 32   # Represents the dimensionality of the feature vectors for each element in the input sequence

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape   # (batch_size, block_size, head_size)
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (C**-0.5)     # Only transpose T & C channel
                                                    # wei = (B, T, head_size) dot (B, T, head_size) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # Avoid attentions to certain elements
        wei = F.softmax(wei, dim=-1)     # Convert into probabilities, on the last dimension of the wei tensor

        out = wei @ v   # Use the calculated attention probabilities (wei) to weigh the values (v)

        return out

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)   # Linear transformation of the outcome of the multi-head layer

    def forward(self, x):
        out = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd),
                                    nn.ReLU(),
                                    nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net.forward(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHead(n_head, head_size, n_embd)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa.forward(self.ln1(x))
        x = x + self.ffwd.forward(self.ln2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        # Initializing the embedding table with size of vocab_size**2
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dimensions)
        self.position_embedding_table = nn.Embedding(block_size, embedding_dimensions)
        # self.head = MultiHead(4, int(embedding_dimensions / 4), embedding_dimensions)     # 4 communication channels (number of self-attention heads)
        #                                                       # , 8 dimensions (head_size)
        # self.net = FeedForward(embedding_dimensions)      # Adding computation on per-node level
        # self.block = Block(embedding_dimensions, n_head=4)    # Single block
        self.blocks = nn.Sequential(Block(embedding_dimensions, n_head=4),
                                    Block(embedding_dimensions, n_head=4),
                                    Block(embedding_dimensions, n_head=4),
                                    nn.LayerNorm(embedding_dimensions)

        )
        self.lm_head = nn.Linear(embedding_dimensions, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)    # (B, T, C), in this case (4 (batch_size), 8 (block_size), 65(n_embd))
                                                     # becase idx is (B, T), and each element from "idx" will go in token_embedding_table
                                                     # and get a row of data (1, 65(C))
        pos_emb = self.position_embedding_table(torch.arange(T))       # (T, C)
        x = tok_emb + pos_emb       # (B, T, C)
        # x = self.head.forward(x)
        # x = self.net.forward(x)
        x = self.blocks.forward(x)
        logits = self.lm_head.forward(x)               # (B, T, vocab_size)

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

    def train(self, train_data, val_data, batch_size, block_size):
        optimizer = torch.optim.AdamW(self.parameters(), lr = learning_rate)
        train_loss = float('inf')
        val_loss = float('inf')

        for iter in range(max_iters):
            # Sample a batch of data
            xb, yb = get_batch(train_data, batch_size, block_size)
            xbv, ybv = get_batch(val_data, batch_size, block_size)

            # Evaluate the loss
            _, train_loss = self.forward(xb, yb)
            _, val_loss = self.forward(xbv, ybv)

            optimizer.zero_grad(set_to_none = True)
            train_loss.backward()
            optimizer.step()

            # Print loss every "eval_interval" steps
            if iter % eval_interval == 0:
                print(f"train_loss: {train_loss.item():.4f}, val_loss: {val_loss.item():.4f}, step: {iter}/{max_iters}")

        print(f"train_loss: {train_loss.item():.4f}, val_loss: {val_loss.item():.4f}, step: {max_iters}/{max_iters}")