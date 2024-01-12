# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          IntervalStrategy, TrainingArguments)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    model: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf")
    dataset: Optional[str] = field( default="mosaicml/dolly_hhrlhf")
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "learning rate"})
    deepspeed: Optional[str] = field(default=None, metadata={"help": "deepspeed config"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "batch size"})
    seq_length: Optional[int] = field(default=2048, metadata={"help": "sequence length"})
    output_dir: Optional[str] = field(default="./temp", metadata={"help": "the output directory"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=20, metadata={"help": "the number of training steps"})
    local_rank: Optional[int] = field(default=-1, metadata={"help": "local rank of process"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
local_rank = script_args.local_rank

# Step 1: Load the dataset
dataset = load_dataset(script_args.dataset)

def process_sample(sample: dict) -> dict:
    sample['text'] = f"### Question: {sample['prompt']}\n ### Answer: {sample['response']}"
    return sample

dataset = dataset.map(process_sample)
print(dataset['train'][0]['text'])

# Step 2: Load the model
device_map = None
quantization_config = None
torch_dtype = None

model = AutoModelForCausalLM.from_pretrained(
    script_args.model,
    quantization_config=quantization_config,
    device_map=device_map,
    torch_dtype=torch_dtype,
    use_cache=True
)

tokenizer = AutoTokenizer.from_pretrained(script_args.model)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    evaluation_strategy=IntervalStrategy.STEPS,
    num_train_epochs=script_args.num_train_epochs,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    deepspeed=script_args.deepspeed,
    max_steps=script_args.max_steps,
    load_best_model_at_end=False,
    report_to='tensorboard',
    local_rank=local_rank,
    save_strategy="steps",
    save_total_limit=2,
    eval_steps=500,
    save_steps=500,
    bf16=True,
)

# Step 4: Define the LoraConfig
peft_config = None

# Step 5: Define the Trainer
collator = DataCollatorForCompletionOnlyLM(" ### Answer:", tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    dataset_text_field='text',
    peft_config=peft_config,
    data_collator=collator
)

trainer.model.resize_token_embeddings(len(tokenizer))
# trainer.train(resume_from_checkpoint=True)
trainer.train()

# Step 6: Save the model
if peft_config is not None:
    trainer.model = trainer.model.merge_and_unload()

trainer.save_model(script_args.output_dir)