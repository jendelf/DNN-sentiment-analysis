# Sentiment Analysis with DistilBERT and TweetEval

## 📌 Objective

The goal of this project is to build a sentiment classification model using a fine-tuned version of `distilbert-base-uncased`.  
The task is to determine whether an expression is a **positive**, **neutral**, or **negative** sentiment.

---

## 📂 Dataset

This project uses the [`benayas/tweet_eval`](https://huggingface.co/datasets/benayas/tweet_eval) dataset from Hugging Face Datasets.
This dataset contains tweets labeled as **positive**, **neutral**, or **negative**, and includes many examples of sarcasm and irony, making it well-suited for training robust sentiment classifiers.

- **Labels**:
  - `0` — negative  
  - `1` — neutral  
  - `2` — positive  
- A **balanced subset** of 3,000 shuffled examples (1,000 per class) is used for training.

---

## 🧠 Model

- **Architecture**: DistilBERT (`distilbert-base-uncased`)
- **Task**: Sequence classification (3 output classes)
- **Tokenizer**: Fast tokenizer with truncation and padding
- **Framework**: Hugging Face Transformers
- **Trainer**: Hugging Face `Trainer` API

The model is fine-tuned on a balanced dataset and saved locally for later use.

---

## 🔁 Workflow

1. Set random seed for reproducibility
2. Load and balance dataset (1000 per class). Then mix up the selected examples for better learning 
3. Tokenize text inputs
4. Define evaluation metric (`accuracy`)
5. Load training configuration from `train_config.json`
6. Fine-tune using Hugging Face's `Trainer`
7. Save fine-tuned model and tokenizer
8. Evaluate on the validation set
9. Make predictions on test inputs
10. Deploy using Gradio
11. Track experiments with MLflow

---

## 🖥️ Gradio Interface

A Gradio app is included for interactive testing. It allows users to:

- Enter a tweet or sentence
- View the predicted sentiment and confidence score

> **Example Output:**  
> `Label: POSITIVE`  
> `Score: 0.9473`

Train the model with:

```bash
python training.py
```

Run it with:

```bash
python gradio_app.py
```

Evaluate on real-world data:

```bash
python evaluate.py
```
