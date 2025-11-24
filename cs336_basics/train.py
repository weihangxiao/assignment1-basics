
from sympy import im
import json
import os
import pathlib
import numpy as np

from tqdm import tqdm
import wandb
import random
import string
import argparse

from cs336_basics.model import Transformer_LM
from cs336_basics.utils import _model_device_and_compile
from cs336_basics.optimizer import AdamW, gradient_clipping, cosine_learning_rate_schedule
from cs336_basics.utils import load_checkpoint, save_checkpoint
from cs336_basics.nn_utils import cross_entropy

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


DATA_DIR = pathlib.Path(__file__).parent.resolve().parent / "data"
CONFIG_PATH = pathlib.Path(__file__).parent.resolve() /'config.json'
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.dat')
VALIDATE_DATA_PATH = os.path.join(DATA_DIR, 'validate.dat')



def data_loader(dataset, batch_size, context_length, device):
    sample_idx = np.random.randint(0, len(dataset) - context_length, size=(batch_size,))
    x = np.array([dataset[idx:idx + context_length] for idx in sample_idx])
    y = np.array([dataset[idx + 1:idx + context_length + 1] for idx in sample_idx])

    device = torch.device(device)

    return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))


def setup_ddp():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))

def cleanup_ddp():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-name', type=str, default=None, help='Run name prefix')
    args_cli = parser.parse_args()
    
    # Setup distributed training if available
    is_ddp = int(os.environ.get('RANK', -1)) != -1
    if is_ddp:
        setup_ddp()
        local_rank = int(os.environ['LOCAL_RANK'])
        device = torch.device(f'cuda:{local_rank}')
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Extract rope_theta and create RotaryPositionalEmbedding
    from cs336_basics.model import RotaryPositionalEmbedding
    model_config = config['model'].copy()
    rope_theta = model_config.pop('rope_theta')
    d_k = model_config['d_model'] // model_config['num_heads']
    rope = RotaryPositionalEmbedding(rope_theta, d_k, model_config['context_length'])
    
    model = Transformer_LM(**model_config, pos_encode=rope, theta=rope_theta)
    params = dict()
    for group in config.values():
        params.update(group)

    class attrDict(dict):
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = attrDict(params)
    
    # Create unique run name with random ID (only on rank 0)
    if local_rank == 0:
        random_id = ''.join(random.choices(string.ascii_lowercase, k=8))
        if args_cli.run_name:
            run_name = f"{args_cli.run_name}_{random_id}"
        else:
            run_name = f"d{args.d_model}_l{args.num_layers}_h{args.num_heads}_{random_id}"
        
        # Initialize wandb only on rank 0
        wandb.init(
            project="cs336-assignment1",
            entity="weihangxiao18-amazon",
            config=params,
            name=run_name
        )
    else:
        run_name = "worker"
    
    # Move model to device
    model = model.to(device)
    
    # Wrap with DDP if distributed
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    
    # Compile if not using DDP
    if not is_ddp and hasattr(torch, 'compile'):
        model = torch.compile(model)


    train_data, validate_data = np.memmap(TRAIN_DATA_PATH, dtype=np.int32, mode='r'), np.memmap(VALIDATE_DATA_PATH, dtype=np.int32, mode='r')

    adamW = AdamW
    optimizer = adamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)
    
    start_iter = 0
    if args.get_checkpoint:
        print(f"resume the checkpoint {args.get_checkpoint}")
        cp_path = pathlib.Path(__file__).parent.resolve().parent / f"checkpoints/cp_iter{args.get_checkpoint}.pt"
        start_iter = load_checkpoint(cp_path, model, optimizer)
        print(f"resume the iterator {start_iter}")
    
    
    
    for iter in tqdm(range(start_iter, args.max_iter), desc='training', disable=local_rank!=0):
        model.train()
        x, y = data_loader(train_data, args.batch_size, args.context_length, device)
        logits = model(x)  # (batch_size, seq_len, vocab_size)
        
        # Reshape for cross entropy: flatten batch and sequence dimensions
        logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_len, vocab_size)
        y = y.view(-1)  # (batch_size * seq_len,)
        
        loss = cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        
        gradient_clipping(model.parameters(), args.clip_grad_norm)
        
        lr = cosine_learning_rate_schedule(iter, args.max_lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        
        # Log training metrics (only rank 0)
        if local_rank == 0 and (iter + 1) % args.log_interval == 0:
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr,
                "iteration": iter + 1
            })
        
        if (iter + 1) % args.valid_interval == 0:
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                x_valid, y_valid = data_loader(validate_data, args.batch_size, args.context_length, device)

                preds = model(x_valid)
                loss = cross_entropy(preds.view(-1, preds.size(-1)), y_valid.view(-1))
                total_loss += loss.item()
        
            if local_rank == 0:
                print(f"iter: {iter + 1}, validate loss: {total_loss:.4f}")
                wandb.log({
                    "val/loss": total_loss,
                    "iteration": iter + 1
                })

            
        
        if local_rank == 0 and (iter + 1) % args.save_interval == 0:
            # Create run-specific checkpoint folder
            run_checkpoint_dir = os.path.join(args.save_path, run_name)
            os.makedirs(run_checkpoint_dir, exist_ok=True)
            cp_name = os.path.join(run_checkpoint_dir, f"iter{iter + 1}.pt")
            # Save unwrapped model if using DDP
            save_model = model.module if is_ddp else model
            save_checkpoint(save_model, optimizer, iter, cp_name)
            print(f"save checkpoint: {cp_name}")
        
        
        


if __name__ == '__main__':
    try:
        main()
    finally:
        if int(os.environ.get('RANK', -1)) != -1:
            cleanup_ddp()
        
        
        