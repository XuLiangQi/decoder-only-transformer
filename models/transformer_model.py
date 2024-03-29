import torch
import torch.nn as nn
from torch.nn import functional as F

from tools.get_batch import get_batch

import yaml

# with open('hyps/hyps-small_model.yaml', 'r') as yaml_file:
#     hyps = yaml.safe_load(yaml_file)
with open('hyps/hyps-large-model.yaml', 'r') as yaml_file:
    hyps = yaml.safe_load(yaml_file)

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(hyps['block_size'], hyps['block_size'])))
        self.dropout = nn.Dropout(hyps['dropout'])

    def forward(self, x):
        B, T, C = x.shape   # (batch_size, block_size, head_size)
        q = self.query(x)   # (B, T, head_size)
        k = self.key(x)     # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)     # Only transpose T & C channel
                                                    # wei = (B, T, head_size) dot (B, T, head_size) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # Avoid attentions to certain elements
        wei = F.softmax(wei, dim=-1)     # Convert into probabilities, on the last dimension of the wei tensor
        wei = self.dropout(wei)

        out = wei @ v   # Use the calculated attention probabilities (wei) to weigh the values (v))

        return out

class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size, n_embd):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)   # Linear transformation of the outcome of the multi-head layer
        self.dropout = nn.Dropout(hyps['dropout'])

    def forward(self, x):
        out = torch.cat([head.forward(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd),
                                    nn.ReLU(),
                                    nn.Linear(4 * n_embd, n_embd),
                                    nn.Dropout(hyps['dropout'])
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
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        # Initializing the embedding table with size of vocab_size**2
        self.token_embedding_table = nn.Embedding(vocab_size, hyps['embedding_dimensions'])
        self.position_embedding_table = nn.Embedding(hyps['block_size'], hyps['embedding_dimensions'])
        # self.head = MultiHead(4, int(embedding_dimensions / 4), embedding_dimensions)     # 4 communication channels (number of self-attention heads)
        #                                                       # , 8 dimensions (head_size)
        # self.net = FeedForward(embedding_dimensions)      # Adding computation on per-node level
        # self.block = Block(embedding_dimensions, n_head=4)    # Single block
        self.blocks = nn.Sequential(*[Block(hyps['embedding_dimensions'], n_head=hyps['n_head']) for _ in range(hyps['n_layers'])])
        self.ln = nn.LayerNorm(hyps['embedding_dimensions'])
        self.lm_head = nn.Linear(hyps['embedding_dimensions'], vocab_size)
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #         if module.bias is not None:
    #             torch.nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        x = self.ln(x)
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
            idx_cond = idx[:, -hyps['block_size']:]
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
