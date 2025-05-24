from typing import List, Tuple

from llama_models.llama3.tokenizer import Tokenizer
from torch import Tensor
from torch.utils.data import Dataset
import polars as pl
import torch
import torch.nn.functional as F


IGNORE_TOKEN: int = -100


class RecipeNERDataset(Dataset):
    _SYSTEM_MSG = "You are a helpful recipe assistant. You are to extract the generic ingredients from each of the recipes provided."
    def __init__(self, df: pl.DataFrame, tokenizer: Tokenizer, device: torch.device):
        super().__init__()
        self._df = df
        self._tokenizer = tokenizer
        self._device = device
        self._special_tokens = tokenizer.special_tokens  # reference

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        title, ingredients, ner = self._df["title"][idx], self._df["ingredients"][idx], self._df["NER"][idx]

        input_tokens: List[int] = [
            self._special_tokens.get("<|begin_of_text|>"),
            self._special_tokens.get("<|start_header_id|>"),
            *self._tokenizer.encode("system", bos=False, eos=False),  # encode returns List[int]
            self._special_tokens.get("<|end_header_id|>"),
            *self._tokenizer.encode("\n\n" + self._SYSTEM_MSG, bos=False, eos=False),
            self._special_tokens.get("<|eot_id|>"),
            self._special_tokens.get("<|start_header_id|>"),
            *self._tokenizer.encode("user", bos=False, eos=False),
            self._special_tokens.get("<|end_header_id|>"),
            *self._tokenizer.encode("\n\n" + self._get_user_msg(title, ingredients), bos=False, eos=False),
            self._special_tokens.get("<|eot_id|>"),
            self._special_tokens.get("<|start_header_id|>"),
            *self._tokenizer.encode("assistant", bos=False, eos=False),
            self._special_tokens.get("<|end_header_id|>"),
            *self._tokenizer.encode("\n\n", bos=False, eos=False),
        ]
        target_tokens: List[int] = [IGNORE_TOKEN] * (len(input_tokens) - 1)

        assistant_tokens = self._tokenizer.encode(ner, bos=False, eos=False) + [self._special_tokens.get("<|eot_id|>")]
        input_tokens.extend(assistant_tokens)
        target_tokens.extend(assistant_tokens)
        target_tokens.extend([self._special_tokens.get("<|end_of_text|>")])

        return (
            torch.tensor(input_tokens, dtype=torch.long, device=self._device),
            torch.tensor(target_tokens, dtype=torch.long, device=self._device)
        )

    @staticmethod
    def _get_user_msg(title: str, ingredients: str) -> str:
        return f"Title: {title}\n\nIngredients: {ingredients}\n\nGeneric ingredients: "


class Collator:
    def __init__(self, pad_id: int):
        self._pad_id = pad_id

    def __call__(self, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        max_token_length = self._get_batch_max_token_length(batch)
        input_tokens_lst, target_tokens_lst = [], []
        for input_tokens, target_tokens in batch:
            padding = max_token_length - input_tokens.shape[0]
            input_tokens = F.pad(input_tokens, (0, padding), value=self._pad_id)
            input_tokens_lst.append(input_tokens)
            target_tokens = F.pad(target_tokens, (0, padding), value=IGNORE_TOKEN)
            target_tokens_lst.append(target_tokens)
        return torch.stack(input_tokens_lst), torch.stack(target_tokens_lst)

    @staticmethod
    def _get_batch_max_token_length(batch: List[Tuple[Tensor, Tensor]]) -> int:
        max_ = 0
        for input_tokens, _ in batch:
            n = input_tokens.shape[0]
            if n > max_:
                max_ = input_tokens.shape[0]
        return max_