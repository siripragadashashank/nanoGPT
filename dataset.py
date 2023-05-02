import os
import numpy as np
import typing as ty

# import config params
from config import *

data_dir = os.path.join(os.path.dirname(__file__), 'data')
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')


def get_batch(split: str) -> ty.Tuple[torch.Tensor, torch.Tensor]:
    """
    returns a batch of input and output tensor of shape (batch_size, block_size)
    :param split: train/val split
    :return: tuple of batched x, y
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    blocks_per_ix = [torch.from_numpy((data[i: i+block_size]).astype(np.int64)) for i in ix]
    next_blocks_per_ix = [torch.from_numpy((data[i+1: i+1+block_size]).astype(np.int64)) for i in ix]
    x = torch.stack(blocks_per_ix, dim=0)
    y = torch.stack(next_blocks_per_ix, dim=0)
    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y


if __name__ == '__main__':
    bx, by = get_batch('train')
    print(bx.shape, by.shape)
