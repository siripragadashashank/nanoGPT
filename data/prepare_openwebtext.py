import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset


def process(example):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out


def encode_openwebtext(fast_dev_run=False):
    num_proc = 4
    dataset = load_dataset("openwebtext")
    split_dataset = dataset['train'].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # do a fast dev run on val dataset
    # if fast_dev_run:
    #     split_dataset.pop('train')

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        # this number is ~4537790, it should be ~9035582489
        # arr_len = np.sum(dset['len'])
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
        total_batches = 1024

        # calculate by iterating over the shards
        arr_len = 0
        for batch_idx in tqdm(range(total_batches), desc=f'calculate size {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr_len += len(arr_batch)
        print(f'calculated size {arr_len}')
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f'idx size {idx}')


if __name__ == '__main__':
    encode_openwebtext(fast_dev_run=True)





