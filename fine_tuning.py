from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import json

# Load GPT-2 model and tokenizer
MODEL_NAME = "gpt2"
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# Add pad token
tokenizer.pad_token = tokenizer.eos_token

# Load and tokenize the dataset
def load_dataset(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()

    # Tokenize each line separately
    encodings = tokenizer(data, truncation=True, padding=True, max_length=512, return_tensors="pt")
    
    return encodings

encodings = load_dataset("healthcare_dataset.json")

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

train_dataset = CustomDataset(encodings)

# Fine-tuning arguments
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Fine-tune
trainer.train()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_healthcare_gpt2")
tokenizer.save_pretrained("fine_tuned_healthcare_gpt2")
