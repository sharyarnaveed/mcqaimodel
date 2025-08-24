# fixed_finetuning.py
"""
Fixed fine-tuning with proper tokenization check
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_from_disk
import torch

# Step 1: Load and VERIFY the tokenized dataset
print("Loading and verifying dataset...")
dataset = load_from_disk("tokenized_biomcq_dataset")

# Check if labels are properly formatted
sample = dataset["train"][0]
labels = sample["labels"]
valid_labels = [x for x in labels if x != -100]

print(f"✓ Dataset loaded: {len(dataset['train'])} train, {len(dataset['test'])} test samples")
print(f"Sample labels length: {len(labels)}")
print(f"Non-padded labels: {len(valid_labels)}")
print(f"Sample valid labels: {valid_labels[:10]}")

if len(valid_labels) == 0:
    print("❌ ERROR: All labels are -100 (padding)! Need to re-tokenize!")
    exit()

# Step 2: Load model and tokenizer
print("Loading model and tokenizer...")
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("✓ Model and tokenizer loaded")

# Step 3: Setup data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Step 4: Setup training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./mcq_model_fixed",
    eval_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=200,
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,  # Disable fp16 for stability
    report_to="none",
    max_grad_norm=1.0,  # Gradient clipping
)

# Step 5: Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

# Step 6: Start training!
print("Starting training...")
print("If loss is not decreasing, we may need to re-tokenize the dataset")

try:
    train_result = trainer.train()
    
    # Save the final model
    trainer.save_model()
    print("✓ Training completed!")
    print("✓ Model saved to './mcq_model_fixed'")
    
except Exception as e:
    print(f"❌ Training failed: {e}")
    print("The issue is likely with the tokenized dataset labels")