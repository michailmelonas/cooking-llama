from llama_models.datatypes import RawMessage
from llama_models.llama3.generation import Llama
import torch


def run(msg: str, max_seq_len: int = 128):
    generator = Llama.build(
        ckpt_dir="/persistent-storage/Llama3.2-1B-Instruct-ft",
        max_seq_len=max_seq_len,
        max_batch_size=1,
        world_size=1,
        device="cuda",
    )
    batch = [
        RawMessage(
            role="system",
            content="You are a helpful recipe assistant. You are to extract the generic ingredients from each of the recipes provided."
        ),
        RawMessage(role="user", content=msg),
    ]
    # todo: resolve {'error': 'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument index in method wrapper_CUDA__index_select)'}
    # next(generator.model.parameters()).device
    # todo OpenAI compatibility
    # todo optimize inference w torch.compile
    with torch.no_grad():
        result = generator.chat_completion(batch)
    return result.generation.content