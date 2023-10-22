import torch
"""
Convert data into batches.

Input:
    data: The data that needs to be batched.
"""
def get_batch(data, batch_size, block_size):
    temp_data = data
    ix = torch.randint(len(temp_data) - block_size, (batch_size,))
    x = torch.stack([temp_data[i:i + block_size] for i in ix])
    y = torch.stack([temp_data[i + 1:i + block_size + 1] for i in ix])  # offset of X
    return x, y