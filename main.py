# Load third-party modules
import torch

# Load custom modules
from tools.txt_loader import load_text


# Read tiny shakespear txt file
text_dir = "data/tinyShakespeare.txt"
text = load_text(text_dir)
# print("Length of dataset in characters: ", len(text))

all_chars = set(text)
all_chars_in_list = list(all_chars)
all_chars_in_list_sorted = sorted(all_chars_in_list)
vocab_size = len(all_chars_in_list_sorted) 
#print(all_chars_in_list_sorted)
#print("Total # vocabs: " + str(vocab_size))

# A simple encoder/decoder
stoi = {}
itos = {}
for i, ch in enumerate(all_chars):
    stoi[ch] = i
    itos[i] = ch

encode = lambda s : [stoi[c] for c in s]
decode = lambda i : [itos[n] for n in i]
# print(encode("Hello World!"))
# print(decode(encode("Hello World!")))

# Convert text into tensor 
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

# Setup a threshold to split data into 90% train, 10% test
train_test_thres = int(0.9 * len(data))
train_data = data[:train_test_thres]
test_data = data[train_test_thres:]

# Break text into chunks
block_size = 8
print(train_data[:block_size + 1])

x = train_data[:block_size]
y = train_data[1:block_size + 1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    print(f"When input is {context} the target is : {target}")