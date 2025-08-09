
import argparse
import logging
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

import pandas as pd

from preprocessing_pipeline_v2 import preprocess_pipeline_v2

# Run preprocessing pipeline first
processed_path = preprocess_pipeline_v2('data/sample_raw_dataset.csv', 'data/v7_sample_train.csv')

# Load the processed data
input_csv = pd.read_csv(processed_path)
input_csv.head(2)

# data_prep_step1.py

import pandas as pd
import json
import re
import warnings

def extract_tag_index(tag: str) -> int:
    """
    Map message tags to an integer index for ordering:
      "<|original|>" → 0
      "<|replyN|>"   → N
    Anything else → large number (sorts last).
    """
    if tag == "<|original|>":
        return 0
    m = re.match(r"<\|reply(\d+)\|>", tag)
    if m:
        return int(m.group(1))
    return 9999

def prepare_pairs(input_df, output_jsonl: str):
    """
    Prepare training pairs from preprocessed email data.
    Now uses the output from preprocess_pipeline_v2 which has columns:
    ['email_id', 'tag', 'subject', 'email_text']
    """
    examples = []

    # Process each thread
    for email_id, group in input_df.groupby("email_id", sort=False):
        grp = group.copy()
        # Sort messages by tag index
        grp['idx'] = grp['tag'].apply(extract_tag_index)
        grp = grp.sort_values('idx')

        msgs = grp['email_text'].tolist()  # Updated column name
        # Grab the subject from the original message row
        subject_rows = grp.loc[grp['idx'] == 0, 'subject']
        subject = subject_rows.iloc[0] if not subject_rows.empty else ""

        # For each reply position i (i >= 1), build one example
        for i in range(1, len(msgs)):
            examples.append({
                "thread": " ".join(msgs[:i]),  # all messages before reply i
                "subject": subject,            # subject from preprocessing
                "email": msgs[i-1],            # immediate predecessor
                "reply": msgs[i],              # this reply
                "tone": "[formal]"             # placeholder tone
            })

    # Write examples to JSONL
    with open(output_jsonl, 'w', encoding='utf-8') as fout:
        for ex in examples:
            fout.write(json.dumps(ex, ensure_ascii=False) + "\n")

output_jsonl = "data/sample_enron_pairs.jsonl"
prepare_pairs(input_csv, output_jsonl)
count = sum(1 for _ in open(output_jsonl, encoding='utf-8'))
print(f"Wrote {count} examples to {output_jsonl}")

# data_prep_step2.py

import json

def build_prompts(input_jsonl: str,
                  output_full: str,
                  output_subject: str):
    """
    Reads each example from input_jsonl (which has keys:
      'thread', 'subject', 'email', 'reply', 'tone')
    and writes two JSONL files, one per prompt format.
    """
    with open(input_jsonl, encoding='utf-8') as fin, \
         open(output_full,    'w', encoding='utf-8') as fout_full, \
         open(output_subject, 'w', encoding='utf-8') as fout_subj:

        for line in fin:
            ex = json.loads(line)

            # 1) Full-thread prompt
            prompt_full = (
                f"{ex['tone']} Thread: {ex['thread']} Reply: {ex['reply']}"
            )
            fout_full.write(
                json.dumps({"text": prompt_full}, ensure_ascii=False) + "\n"
            )

            # 2) Subject + last email prompt
            prompt_subj = (
                f"{ex['tone']} Subject: {ex['subject']} ┃ "
                f"Last message: {ex['email']} Reply: {ex['reply']}"
            )
            fout_subj.write(
                json.dumps({"text": prompt_subj}, ensure_ascii=False) + "\n"
            )

input_jsonl    = "data/sample_enron_pairs.jsonl"
output_full    = "data/sample_enron_prompts_full.jsonl"
output_subject = "data/sample_enron_prompts_subject.jsonl"

build_prompts(input_jsonl, output_full, output_subject)
print("Built prompts")

# data_prep_step3.py

import os
from datasets import load_dataset
from transformers import AutoTokenizer

def tokenize_and_save(input_jsonl: str, output_dir: str, max_len: int = 512):
    """
    1) Loads the JSONL at `input_jsonl` (expects one {"text": ...} per line).
    2) Tokenizes to GPT-2 input_ids (truncates/pads to max_len).
    3) Sets labels = input_ids for causal LM.
    4) Saves the processed dataset to `output_dir` (creates it if needed).
    """
    # a) Load the prompts as a Hugging Face Dataset
    ds = load_dataset("json", data_files=input_jsonl, split="train")

    # b) Load GPT-2 tokenizer and set pad token
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # c) Tokenization function (batched)
    def tokenize_fn(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length"
        )
        # Use the inputs themselves as labels for next‐token prediction
        enc["labels"] = enc["input_ids"].copy()
        return enc

    # d) Apply it
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    # e) Save to disk
    os.makedirs(output_dir, exist_ok=True)
    tokenized.save_to_disk(output_dir)
    print(f"> Saved {len(tokenized)} examples to '{output_dir}'")

full_in  = "data/sample_enron_prompts_full.jsonl"
subj_in  = "data/sample_enron_prompts_subject.jsonl"
full_out = "data/tokenized_full"
subj_out = "data/tokenized_subject"

tokenize_and_save(full_in, full_out)
tokenize_and_save(subj_in, subj_out)
print("Tokenization complete")

#!/usr/bin/env python
# train_step4_v2.py

import os
import logging
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fine_tune(tokenized_dir, output_dir):
    # 1) Load and split
    ds = load_from_disk(tokenized_dir)
    ds = ds.train_test_split(test_size=0.1) if not isinstance(ds, dict) else ds
    train_ds = ds.get("train", ds.get("all"))
    eval_ds  = ds.get("test",  ds.get("validation"))

    logger.info(f"→ {tokenized_dir}: train={len(train_ds)}, eval={len(eval_ds)}")

    # 2) Model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # 3) Inject LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8, lora_alpha=32, lora_dropout=0.1,
    )
    model = get_peft_model(model, lora_cfg)

    # 4) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 5) Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        report_to="none",
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 7) Train & save
    logger.info(f"✨ Fine‐tuning {tokenized_dir} → {output_dir}")
    trainer.train()
    trainer.save_model(output_dir)
    logger.info(f"✔ Done {output_dir}\n")

pairs = [
    ("data/tokenized_full",    "outputs/gpt2_lora_full"),
    ("data/tokenized_subject", "outputs/gpt2_lora_subject"),
]
for tok_dir, out_dir in pairs:
    os.makedirs(out_dir, exist_ok=True)
    fine_tune(tok_dir, out_dir)



from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "outputs/gpt2_lora_full"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

prompt = "Subject: Meeting follow-up\n\nHi team,"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

prompt = """Subject: Weekly Team Update

Hi everyone,
The client feedback for Project Atlas was very positive. Great work!
For those interested, we’ll do a quick walkthrough of the new CRM interface during Friday’s stand-up.
Thanks!
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate output
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))