# cooking-llama
##### Fine-tuning Llama-3.2-1B-Instruct on the RecipesNLG dataset

<img src="cooking-llama.png" width="250" style="display: block; margin: 0 auto" />

Following https://cookbook.openai.com/examples/how_to_finetune_chat_models, we fine-tune our own instance of Llama-3.2-1B-Instruct.
Our approach is purely `torch` based. Model weights were obtained from https://www.llama.com/models/llama-3/.
See `./train` and `./inference` for training and inference logic, respectively.

<div style="display: flex; justify-content: center; gap: 20px; margin: 40px 0;">
  <img src="step_train_loss.png" width="200" />
  <img src="avg_val_loss.png" width="200" />
</div>

