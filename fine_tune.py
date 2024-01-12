import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

dataset = load_dataset("mosaicml/dolly_hhrlhf", split="train")

model_size = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(f"meta-llama/Llama-2-{model_size}-chat-hf")
tokenizer = AutoTokenizer.from_pretrained(f"meta-llama/Llama-2-{model_size}-chat-hf")

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Question: {example['prompt'][i]}\n ### Answer: {example['response'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)

trainer.train()