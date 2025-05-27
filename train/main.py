from pathlib import Path
import shutil

import polars as pl
import torch

from download_weights import download_weights
from train import get_fine_tuned_model


BASE_CKPT_DIR = Path("/persistent-storage/Llama3.2-1B-Instruct")
FT_CKPT_DIR = Path("/persistent-storage/Llama3.2-1B-Instruct-ft")


def run(steps: int, batch_size: int):
    # download base weights from s3 to /persistent-storage
    if BASE_CKPT_DIR.exists() is False:
        download_weights()

    # load training data and fine-tune weights
    df = pl.read_csv("/persistent-storage/cookbook_recipes_nlg_10k.csv")
    # todo: implement DDP
    model = get_fine_tuned_model(BASE_CKPT_DIR, df, steps, batch_size, 0.0001, 1)

    # save weights
    FT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), FT_CKPT_DIR / "consolidated.00.pth")
    shutil.copy(BASE_CKPT_DIR / "params.json", FT_CKPT_DIR / "params.json")
    shutil.copy(BASE_CKPT_DIR / "checklist.chk", FT_CKPT_DIR / "checklist.chk")

    return "ok"
