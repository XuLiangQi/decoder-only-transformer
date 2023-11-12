import torch
import torch.nn as nn
from torch.nn import functional as F

from tools.get_batch import get_batch

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        # Initializing the embedding table with size of vocab_size**2
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, X, Y=None):
        # idx and targets are both (B, T) tensor of integers
        logits = self.token_embedding_table(X)    # (B, T, C), in this case (4 (batch_size), 8 (block_size), 65(vocab_size))
                                                     # becase idx is (B, T), and each element from "idx" will go in token_embedding_table
                                                     # and get a row of data (1, 65(C))          # (B, T, vocab_size)

        if Y is not None:
            B, T, C = logits.shape

            logits = logits.view(B*T, C)        # Returns a new tensor with the same data as the self tensor but of a different shape.
            targets = Y.view(B*T)

            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, _ = self.forward(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :]   # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1)     # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples = 1)    # (B, 1)
            # Append sampled index to the running sequencing
            idx = torch.cat((idx, idx_next), dim = 1)      # (B, T+1)

        return idx

