from pathlib import Path
from typing import Tuple
import json
import os
import random

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from llama_models.llama3.args import ModelArgs
from llama_models.llama3.tokenizer import Tokenizer
from torch.utils.data import DataLoader, RandomSampler
import polars as pl
import torch
import torch.nn as nn
import wandb

from dataset import Collator, IGNORE_TOKEN, RecipeNERDataset
from model import CustomTransformer


def get_fine_tuned_model(
    ckpt_dir: Path,
    df: pl.DataFrame,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int = 1
) -> CustomTransformer:
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project="cooking-llama",
        config={"learning_rate": lr, "batch_size": batch_size, "steps": steps}
    )

    # torch setup + instantiate model (lifted from https://github.com/meta-llama/llama-models/tree/main)
    device = torch.device("cuda")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    torch.manual_seed(seed)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if not model_parallel_is_initialized():
        initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    checkpoints = sorted(ckpt_dir.glob("*.pth"))
    ckpt_path = checkpoints[get_model_parallel_rank()]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    with open(ckpt_dir / "params.json", "r") as f:
        params = json.loads(f.read())
    model_args: ModelArgs = ModelArgs(
        max_seq_len=512,
        max_batch_size=batch_size,
        **params,
    )
    tokenizer = Tokenizer.get_instance()
    assert model_args.vocab_size == tokenizer.n_words

    torch.set_default_device(device)
    if torch.cuda.is_bf16_supported():
        torch.set_default_dtype(torch.bfloat16)
    else:
        torch.set_default_dtype(torch.half)

    model = CustomTransformer(model_args)
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)

    # data setup
    train_df, val_df = get_train_and_val_dfs(df, seed)
    train_dataset = RecipeNERDataset(train_df, tokenizer, device)
    val_dataset = RecipeNERDataset(val_df, tokenizer, device)
    eval_train_dataset = RecipeNERDataset(train_df.head(len(val_dataset)), tokenizer, device)  # estimate train loss

    collator = Collator(tokenizer.pad_id)
    train_sampler = RandomSampler(
        train_dataset, replacement=True, num_samples=steps*batch_size, generator=torch.Generator(device="cuda")
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*4, collate_fn=collator, drop_last=True)
    eval_train_loader = DataLoader(eval_train_dataset, batch_size=batch_size*4, collate_fn=collator, drop_last=True)

    # training
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_TOKEN)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step, (input_tokens, target_tokens) in enumerate(train_loader):
        if step % 100 == 0:
            avg_val_loss = get_avg_loss(model, val_loader, loss_fn)
            avg_train_loss = get_avg_loss(model, eval_train_loader, loss_fn)
            wandb.log({
                "step": step,
                "avg_val_loss": avg_val_loss,
                "avg_train_loss": avg_train_loss
            })

        optimizer.zero_grad()
        output = model(input_tokens)
        b, t, c = output.shape
        loss = loss_fn(output.view(b * t, c), target_tokens.view(b * t))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        wandb.log({"step": step, "step_train_loss": loss.item()})

    wandb.finish()
    return model


def get_avg_loss(model: CustomTransformer, loader: DataLoader, loss_fn: nn.CrossEntropyLoss) -> float:
    """Computes average of mean loss per batch."""
    total_loss = 0.0
    with torch.no_grad():
        for input_tokens, target_tokens in loader:
            output = model(input_tokens)
            b, t, c = output.shape
            loss = loss_fn(output.view(b*t, c), target_tokens.view(b*t))
            total_loss += loss.item()

    return total_loss / len(loader)


def get_train_and_val_dfs(df: pl.DataFrame, seed: int, train_proportion: float=0.9) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Randomly split full df into train/val dfs."""
    idxs = list(range(len(df)))
    random.seed(seed)
    random.shuffle(idxs)
    n_train = int(train_proportion * len(df))
    train_idxs, val_idxs = idxs[:n_train], idxs[n_train:]
    return df[train_idxs], df[val_idxs]
