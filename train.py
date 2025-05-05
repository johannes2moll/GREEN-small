import datasets
import pandas as pd
import transformers
import torch

train_datapath = "jomoll/GREEN-V3"
eval_datapath = "jomoll/GREEN-V3"
max_length = 512
batch_size = 128
model_name = "FacebookAI/roberta-base"

model = transformers.RobertaForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = transformers.RobertaTokenizer.from_pretrained(model_name)

def load_and_preprocess_dataset(data_path, tokenizer, max_len, split="train", system_message="", batch_size=batch_size):
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
    input_text = ["Reference report:\n"+reference_report+"\n Candidate report:\n"+candidate_report 
                  for reference_report, candidate_report in zip(originals, candidates)]

    # Use padding="longest" to only pad to the longest sequence in the batch for efficiency
    inputs = tokenizer(input_text, padding="longest", truncation=True, max_length=max_len, return_tensors="pt")
    
    # Using the tokenizer's padding, no need to add the attention mask manually
    inputs["labels"] = green_scores

    return inputs

# Load and preprocess datasets
train_dataset = load_and_preprocess_dataset(train_datapath, tokenizer, split="train", max_len=max_length)
eval_dataset = load_and_preprocess_dataset(eval_datapath, tokenizer, split="validation", max_len=max_length)

# Set training arguments
training_args = transformers.TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # You can switch to "epoch" if preferred
    #eval_steps=500,  # Evaluate every 500 steps
    save_strategy="epoch",  # Save checkpoint after every epoch
    learning_rate=1e-4,  # You can experiment with 1e-4 or 2e-5
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4, 
    num_train_epochs=1,
    weight_decay=0.01,
    report_to="wandb",  # Reporting to wandb, ensure you're logged in if using it
    logging_dir="./logs",  # Log directory for tensorboard
    logging_steps=500,  # Log every 500 steps
    load_best_model_at_end=True,  # Automatically load the best model after training
    metric_for_best_model="eval_loss",  # You can track eval_loss or another metric
)

# Trainer setup
trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training
trainer.train()

# push to hub
trainer.push_to_hub(repo_path_or_name="jomoll/roberta-green-1", 
                     commit_message="Initial commit", 
                     blocking=True, 
                     auto_lfs_prune=True)
# Save the model and tokenizer
trainer.save_model("./results")
tokenizer.save_pretrained("./results")
