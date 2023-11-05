# Load third-party modules
import torch
from torch.nn import functional as F

# Load custom modules
from tools.txt_loader import load_text
from tools.get_batch import get_batch
from models.bigram_language_model import BigramLanguageModel as BLM
from models.transformer_model import TransformerModel as TM

import yaml

# with open('hyps/hyps-small_model.yaml', 'r') as yaml_file:
#     hyps = yaml.safe_load(yaml_file)
with open('hyps/hyps-large_model.yaml', 'r') as yaml_file:
    hyps = yaml.safe_load(yaml_file)

batch_size = hyps['batch_size']  # How many independent sequences will we precess in parallel
block_size = hyps['block_size']  # What is the maximum context length for predictions
max_tokens = 500

# Check and assign GPU (CUDA) or MPS (Apple Metal) if available
if torch.cuda.is_available():
    torch.set_default_device('cuda')
elif torch.backends.mps.is_available():
    # Use CPU due to bug
    pass
    # torch.set_default_device("mps")

# Read tiny shakespear txt file
text_dir = "data/tinyShakespeare.txt"
text = load_text(text_dir)
# print("Length of dataset in characters: ", len(text))

all_chars = set(text)
all_chars_in_list = list(all_chars)
all_chars_in_list_sorted = sorted(all_chars_in_list)
vocab_size = len(all_chars_in_list_sorted)

# A simple encoder/decoder
stoi = {}
itos = {}
for i, ch in enumerate(all_chars):
    stoi[ch] = i
    itos[i] = ch

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])   # ''.join connects tuples with '' and convert them into string


# Convert text into tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Set up a threshold to split data into 90% train, 10% test
train_test_thres = int(0.9 * len(data))
train_data = data[:train_test_thres]
val_data = data[train_test_thres:]

torch.manual_seed(1337)

model = TM(vocab_size)
# idx = torch.zeros((1, 1), dtype = torch.long)
print(decode(model.generate(torch.zeros((1, 1), dtype = torch.long), max_new_tokens=max_tokens)[0].tolist()))
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyps['eval_iters'])
        for k in range(hyps['eval_iters']):
            if split =='train':
                X, Y = get_batch(train_data, hyps['batch_size'], hyps['block_size'])
            else:
                X, Y = get_batch(val_data, hyps['batch_size'], hyps['block_size'])
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    optimizer = torch.optim.AdamW(model.parameters(), lr=hyps['learning_rate'])

    for iter in range(hyps['max_iters']):
        # Sample a batch of data
        xb, yb = get_batch(train_data, batch_size, block_size)

        # Evaluate the loss
        _, train_loss = model.forward(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        optimizer.step()

        if iter % hyps['eval_interval'] == 0 or iter == hyps['max_iters'] - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

train()
print(decode(model.generate(torch.zeros((1, 1), dtype = torch.long), max_new_tokens=max_tokens)[0].tolist()))