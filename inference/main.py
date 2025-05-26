from llama_models.datatypes import RawMessage
from llama_models.llama3.generation import Llama
import torch


generator = Llama.build(
    ckpt_dir="/persistent-storage/Llama3.2-1B-Instruct-ft",
    max_seq_len=128,
    max_batch_size=1,
    world_size=1,
    device="cuda",
)


def run(msg: str):
    batch = [
        RawMessage(
            role="system",
            content="You are a helpful recipe assistant. You are to extract the generic ingredients from each of the recipes provided."
        ),
        RawMessage(role="user", content=msg),
    ]
    # todo OpenAI compatibility
    with torch.no_grad():
        result = generator.chat_completion(batch)
    print(result)
    return 200