import torch

def get_batch(split, train_data, val_data, batch_size, block_size):
    temp_data = train_data if split == 'train' else val_data
    ix = torch.randint(len(temp_data) - block_size, (batch_size,))
    x = torch.stack([temp_data[i:i + block_size] for i in ix])
    y = torch.stack([temp_data[i + 1:i + block_size + 1] for i in ix])
    return x, y