import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import gradio as gr

# ===================== 3. LOAD THE TOKENIZER AND MODEL =====================
print("Step 3: Loading the fine-tuned model and tokenizer...")
model_path = "./ner_finetuned_model"  
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# ===================== 12. TEST THE MODEL =====================
print("Step 12: Testing the model on a sample...")
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

# Start Gradio UI
iface.launch(share=True)
