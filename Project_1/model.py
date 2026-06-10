import os
import pandas as pd
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          DataCollatorForSeq2Seq)

script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "train.csv")
test_path = os.path.join(script_dir, "test.csv")

if not os.path.isfile(train_path):
    raise FileNotFoundError(f"Training data file not found: {train_path}")
if not os.path.isfile(test_path):
    raise FileNotFoundError(f"Test data file not found: {test_path}")

MODEL_NAME = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
train_ds = Dataset.from_pandas(train_df)
test_ds = Dataset.from_pandas(test_df)

MAX_INPUT  = 512
MAX_TARGET = 128

def preprocess(batch):
    inputs = tokenizer(batch["findings"],
                       max_length=MAX_INPUT, truncation=True, padding="max_length")
    targets = tokenizer(batch["impression"],
                        max_length=MAX_TARGET, truncation=True, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

train_ds = train_ds.map(preprocess, batched=True)
test_ds  = test_ds.map(preprocess, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

args = Seq2SeqTrainingArguments(
    output_dir="./model_output",
    num_train_epochs=3,           # increase to 5 if score is below 0.3
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    predict_with_generate=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                    # set True if you have a GPU
    dataloader_pin_memory=False    # disable pin memory on CPU to avoid warning
)

collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=collator
)

trainer.train()
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")