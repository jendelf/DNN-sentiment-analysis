import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "./ner_finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

test_df = pd.read_csv("data/real_data.csv") 

test_df.columns = test_df.columns.str.strip()

test_df['combined_text'] = test_df.iloc[:, 1:].apply(lambda x: ' '.join(x.astype(str)), axis=1)


texts = test_df['combined_text'].tolist() 


inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")


model.eval()
with torch.no_grad(): 
    outputs = model(**inputs)


predictions = torch.argmax(outputs.logits, dim=-1)


labels = ["negative",
        "neutral",
        "positive"]  

predicted_labels = [labels[prediction] for prediction in predictions]

test_df['prediction'] = predicted_labels 
test_df.to_csv("result_file.csv", index=False)

