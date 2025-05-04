import datasets
import pandas as pd
import transformers
import torch

train_datapath = "jomoll/GREEN-V3"
eval_datapath = "jomoll/GREEN-V3"
max_length = 512

model_name = "FacebookAI/roberta-base"

model = transformers.RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

def load_and_preprocess_dataset(data_path, tokenizer, max_len, split="train", system_message="", batch_size=16):
    """Load and preprocess dataset in batches."""
    # Load hf dataset
    raw_data = datasets.load_dataset(data_path, split=split)

    # Preprocess data in batches
    processed_data = raw_data.map(
        lambda batch: preprocess_batch(batch, tokenizer, max_len, system_message),
        batched=True,
        batch_size=batch_size,
        desc="Running tokenizer on "+split+" dataset")

    return processed_data

def preprocess_batch(batch, tokenizer: transformers.PreTrainedTokenizer, max_len: int, system_message: str = ""):
    """Preprocess a batch of samples."""
    # Validate fields
    originals = batch.get("reference", [])
    candidates = batch.get("candidate", [])
    green_scores = batch.get("green_score", [])

    # Tokenize the input and target text
    input_text = ["Reference report:\n"+reference_report+"\n Candidate report:\n"+candidate_report for reference_report, candidate_report in zip(originals, candidates)]

    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)  # Add attention mask

    targets = green_scores
    inputs["labels"] = targets

    return inputs

train_dataset = load_and_preprocess_dataset(train_datapath, tokenizer, split="train", max_len=max_length)
eval_dataset = load_and_preprocess_dataset(eval_datapath, tokenizer, split="validation", max_len=max_length)

# train the model
trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    args=transformers.TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="epoch",
        learning_rate=1e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    ),
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
# Save the model
trainer.save_model("./results")
# Save the tokenizer
tokenizer.save_pretrained("./results")
