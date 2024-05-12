from unsloth import FastLanguageModel
from transformers import AutoTokenizer
import torch
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

tokenizer.model_max_length = 2048
tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = 'right'

"""We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

import pandas as pd
from sklearn.model_selection import train_test_split
import datasets

train = datasets.load_dataset("grammarly/coedit", split = "train").to_pandas()
val = datasets.load_dataset("grammarly/coedit", split = "validation").to_pandas()

data = pd.concat([train, val])
data[['instruction', 'input']] = data['src'].str.split(': ', n=1, expand=True)
data = data.rename(columns={"tgt": "output"})
data = data.drop(columns=["_id", "src"])

stratify_col = data['task']

# train_df, test_df = train_test_split(
#     data,
#     test_size=0.2,
#     random_state=42,
#     stratify=stratify_col
# )

# print(train_df['task'].value_counts(normalize=True))
# print(test_df['task'].value_counts(normalize=True))

def formatting_prompts_func(examples, tokenizer):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        message = [
            {"role": "user", "content": instruction + ": " + input},
            {"role": "assistant", "content": output},
        ]
        text = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return { "text" : texts, }

train_ds = datasets.Dataset.from_pandas(data)
# test_ds = datasets.Dataset.from_pandas(test_df)

train_ds = train_ds.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, batched = True,)
# test_ds = test_ds.map(formatting_prompts_func, fn_kwargs={"tokenizer": tokenizer}, batched = True,)

print(train_ds[0]['text'])

from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    # eval_dataset = test_ds,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 10,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs=2,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        save_steps=100,
        save_total_limit=4,  # Limit the total number of checkpoints
        evaluation_strategy="no",
#         eval_steps=1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        load_best_model_at_end=True,
        save_strategy="no",
    ),
)

#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print("Saving model to local")
model.save_pretrained("coedit-Phi-3-mini-4k-instruct-fulltrain-2") # Local saving
tokenizer.save_pretrained("coedit-Phi-3-mini-4k-instruct-fulltrain-2")

print("Saving model to hub")
model.push_to_hub("letheviet/coedit-Phi-3-mini-4k-instruct-fulltrain-2", token = "") # Online saving
tokenizer.push_to_hub("letheviet/coedit-Phi-3-mini-4k-instruct-fulltrain-2", token = "") # Online saving

#trainer.evaluate()
