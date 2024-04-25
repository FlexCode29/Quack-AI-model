import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, PushToHubCallback
from huggingface_hub import notebook_login
from huggingface_hub import HfFolder

# Log in to Hugging Face
HfFolder.save_token('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')

# Initialize wandb
wandb.init(project='model_finetuning')

# Load the dataset
dataset = load_dataset("ArtifactAI/arxiv_python_research_code")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("budecosystem/code-millenials-1b")
model = AutoModelForCausalLM.from_pretrained("budecosystem/code-millenials-1b")

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['code'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    max_steps=11502,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="adamw",
    fp16=True,
    report_to="wandb",
    push_to_hub=True,
    hub_strategy="every_save"
)

# Initialize Trainer with PushToHubCallback for automatic model pushing
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    callbacks=[PushToHubCallback(output_dir="./results", tokenizer=tokenizer)]
)

# Start training
trainer.train()
