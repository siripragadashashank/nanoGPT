import torch

out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = True
wandb_project = 'nanoGPT'
wandb_run_name = 'gpt2'

# dataset
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# model
n_layers = 12
n_heads = 12
embed_size = 768
dropout = 0.0
bias = False

# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# device
device = 'cuda'
dtype = 'bfloat16'
compile = True
device_type = 'cuda'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

master_process = True
seed_offset = 0
ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")




