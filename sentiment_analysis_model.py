import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import csv
import mlflow
import json
import gradio as gr
import random
import torch
from datasets import Dataset


# ===================== 0. creating random seed =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  

set_seed(42)  

# ===================== 1. LOAD THE DATASET =====================
print("Step 1: Loading the tweet_eval dataset...")
dataset = load_dataset("tweet_eval", "sentiment")
print(f"Dataset loaded with {len(dataset['train'])} training examples")
with open('tweet_eval_dataset.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"]) 
    for example in dataset["train"]:
        writer.writerow([example["text"], example["label"]])

# ===================== 2. DEFINE LABEL MAPPING =====================
print("Step 2: Setting up the label mapping...")
label_list = [
    "negative",
    "neutral",
    "positive"
]

label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# ===================== 2.1 splitting the data into the classes =====================
positive_examples = [example for example in dataset['train'] if example['label'] == 2]
neutral_examples = [example for example in dataset['train'] if example['label'] == 1 ]
negative_examples = [example for example in dataset['train'] if example['label'] == 0]

# ===================== 2.2 getting only 1000 examples of each class and mix them =====================
balanced_positive = positive_examples[:1000]
balanced_neutral = neutral_examples[:1000]
balanced_negative = negative_examples[:1000]
balanced_dataset = balanced_positive + balanced_neutral + balanced_negative
random.shuffle(balanced_dataset)
print(f"Balanced dataset size: {len(balanced_dataset)}")
balanced_dataset = Dataset.from_dict({
    'text': [example['text'] for example in balanced_dataset],
    'label': [example['label'] for example in balanced_dataset]
})

# ===================== 3. LOAD THE TOKENIZER AND MODEL =====================
print("Step 3: Loading the base model and tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

output_dir = "./ner_original_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Original model and tokenizer saved to {output_dir}")

# ===================== 4. TOKENIZE THE DATA =====================
print("Step 4: Tokenizing the dataset...")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding=True, truncation=True)

# Tokenize both train and eval datasets
train_dataset = balanced_dataset.map(tokenize_function, batched=True)
eval_dataset = dataset['validation'].select(range(1000)).map(tokenize_function, batched=True)

# ===================== 5. SETUP TRAINING ARGUMENTS =====================
with open('train_config.json', 'r') as json_file:
    config = json.load(json_file)

training_args = TrainingArguments(
    output_dir=config["output_dir"],  
    evaluation_strategy=config["eval_strategy"],  
    learning_rate=config["learning_rate"],  
    per_device_train_batch_size=config["per_device_train_batch_size"],  
    per_device_eval_batch_size=config["per_device_eval_batch_size"],  
    num_train_epochs=config["num_train_epochs"],  
    weight_decay=config["weight_decay"],  
    save_strategy=config["save_strategy"],  
    load_best_model_at_end=config["load_best_model_at_end"],  
    metric_for_best_model="eval_accuracy", 
    greater_is_better=True,  
    push_to_hub=config["push_to_hub"],
    fp16=True
)

# ===================== 6. DEFINE METRICS =====================
print("Step 6: Setting up evaluation metrics...")
metric = evaluate.load("accuracy")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1) 
    results = metric.compute(predictions=predictions, references=labels)
    return results

# ===================== 7. CREATE DATA COLLATOR =====================
print("Step 7: Creating data collator...")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ===================== 8. SET UP TRAINER =====================
print("Step 8: Setting up the Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ===================== 9. START TRAINING =====================
print("Step 9: Starting the training process...")
trainer.train()

# ===================== 10. EVALUATE THE MODEL =====================
print("Step 10: Evaluating the fine-tuned model...")
final_results = trainer.evaluate()
print(f"Final evaluation results: {final_results}")

# ===================== 11. SAVE THE MODEL LOCALLY =====================
print("Step 11: Saving the fine-tuned model...")
output_dir = "./ner_finetuned_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model and tokenizer saved to {output_dir}")

# ===================== 12. TEST THE MODEL =====================
print("Step 12: Testing the model on a sample...")
from transformers import pipeline


# Load the fine-tuned model
sentiment_analysis_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Test on a sample
sample_text = "Mark Watney visited Microsoft headquarters in Seattle last month."
results = sentiment_analysis_pipeline(sample_text)

print("\nModel predictions on test sample:")
for result in results:
    print(f"Label: {result['label']}, Score: {result['score']:.4f}")

# ===================== 13. DEPLOY WITH GRADIO =====================

print("Step 13: Deploying model with Gradio...")

def predict_sentiment(text):
    results = sentiment_analysis_pipeline(text)
    label = results[0]['label']
    score = results[0]['score']
    return f"label: {label}, score: {score:.4f}"

# creating interface
iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter text..."),
    outputs=gr.Textbox(),
    title="Sentiment Analysis",
    description="Enter text and the model will determine its sentiment (positive, negative, neutral)."
)

iface.launch()

mlflow.set_experiment("sentiment_analysis_finetuning") 
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")