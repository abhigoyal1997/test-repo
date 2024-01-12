import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

dataset = load_dataset("mosaicml/dolly_hhrlhf", split="train")

model_size = sys.argv[1]
batch_size = sys.argv[2]

model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{model_size}-chat-hf")
tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-{model_size}-chat-hf")
tokenizer.pad_token = tokenizer.eos_token

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['response'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir="temp/",
    per_device_train_batch_size=batch_size,
    gradient_checkpointing=False,
    num_train_epochs=1,
    learning_rate=1e-5,
    logging_steps=1,
    deepspeed='./deepspeed_config.json',
    max_steps=20,
    load_best_model_at_end=True,
    report_to='tensorboard',
    local_rank=0,
    save_strategy="steps",
    save_total_limit=2,
    eval_steps=500,
    save_steps=500,
    bf16=True,
)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()