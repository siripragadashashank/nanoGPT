import os
import tiktoken
import numpy as np


def encode_ss(input_path='input.txt'):
    ss_path = input_path
    with open(ss_path, 'r') as ss:
        data = ss.read()

    n = len(data)

    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    encoder = tiktoken.get_encoding("gpt2")

    train_ids = encoder.encode_ordinary(train_data)
    val_ids = encoder.encode_ordinary(val_data)

    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile('ss_train.bin')
    val_ids.tofile('ss_val.bin')


if __name__ == '__main__':
    encode_ss()
