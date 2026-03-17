import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

MODEL_NAME = "google/flan-t5-small"
DATA_FILE = "training_data.jsonl"
MODEL_OUTPUT_DIR = "./models/t5_jargon_v1"

print(f"Loading tokenizer for {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    inputs = ["Translate to corporate jargon: " + text for text in examples["input"]]
    targets = [text for text in examples["output"]]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    
    processed_labels = []
    for label_row in labels["input_ids"]:
        processed_row = [(l if l != tokenizer.pad_token_id else -100) for l in label_row]
        processed_labels.append(processed_row)
    
    model_inputs["labels"] = processed_labels
    return model_inputs

print(f"Loading and processing dataset from {DATA_FILE}...")
raw_dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Use ALL data for training (no split)
train_dataset = raw_dataset
print(f"Total training samples: {len(train_dataset)}")

tokenized_train = train_dataset.map(preprocess_function, batched=True)

print(f"Loading base model {MODEL_NAME}...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Aggressive training configuration
total_steps = (len(train_dataset) // 4) * 20
warmup_steps = int(0.15 * total_steps)

training_args = TrainingArguments(
    output_dir=f"{MODEL_OUTPUT_DIR}_checkpoints",
    per_device_train_batch_size=4,
    num_train_epochs=20,
    learning_rate=5e-4,
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3,
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Starting model training...")
train_result = trainer.train()
print("Training complete.")

final_loss = train_result.metrics['train_loss']
print(f"\nFinal Training Loss: {final_loss:.4f}")
if final_loss < 1.0:
    print("✓ SUCCESS: Model converged properly!")
else:
    print("⚠ WARNING: Loss still high. Consider training longer.")

print(f"\nSaving model and tokenizer to {MODEL_OUTPUT_DIR}...")
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
trainer.save_model(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print(f"Model v1 saved successfully to {MODEL_OUTPUT_DIR}")
