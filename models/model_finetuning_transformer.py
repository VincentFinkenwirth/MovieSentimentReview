
import re
import random
import numpy as np
import torch
from tools.data_combination import load_data
import pandas as pd
# Hugging Face libraries
from datasets import Dataset
from transformers import BertTokenizerFast,BertForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,pipeline, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow
import time
from tools.viszualizer import plot_confusion_matrix
import os


# GLOBAL
DATA_PATH = "../data/preprocessed/split_data_v2"
MLFLOW_PATH = "file:///C:/python/SpamDetection/mlruns"

# Set mlflow tracking directory
mlflow.set_tracking_uri(MLFLOW_PATH)

# Set Random Seeds for Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # (Optional) Force certain PyTorch operations to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Define a Preprocessor to Remove HTML Tags

def remove_html_tags(text):
    cleaned_text = re.sub(r"<[^>]*>", "", text)
    return cleaned_text

def preprocess_function(data):
    cleaned_texts = [remove_html_tags(text) for text in data["text"]]
    return {"text": cleaned_texts}



def load_data_hug(data_path):
    # Load data from file
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_path, val_set=True, X_col="text")
    # Combine the X and y while flattening X
    train_df = pd.DataFrame({"text": np.array(X_train).flatten(), "label": y_train})
    val_df = pd.DataFrame({"text": np.array(X_val).flatten(), "label": y_val})
    test_df = pd.DataFrame({"text": np.array(X_test).flatten(), "label": y_test})
    # Turn into Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # apply html removal
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    return train_dataset, val_dataset, test_dataset




def main():
    mlflow.set_experiment(f"bert-imdb_{time.strftime('%Y-%m-%d-%H-%M-%S')}_v2")
    # Set the seed
    set_seed(42)
    with mlflow.start_run(run_name=f"bert-imdb_{time.strftime('%Y-%m-%d-%H-%M-%S')}"):
        mlflow.autolog()
        # Tokenization
        # Use BERT tokenizer (uncased)
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512
            )

        # Load the data
        train_dataset, val_dataset, test_dataset = load_data_hug(DATA_PATH)

        # Tokenize the data
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)
        test_dataset = test_dataset.map(tokenize_function, batched=True)



        # We set the format for PyTorch Tensors
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # Initialize the Model
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2  # positive or negative
        )


        # Preparation
        # Data Collator for Padding
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir="./bert-imdb-model",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=2,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            load_best_model_at_end=True,  # save best checkpoint
            seed=42
        )

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, preds),
                "f1": f1_score(labels, preds),
                "precision": precision_score(labels, preds, zero_division=0),
                "recall": recall_score(labels, preds)
            }

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        # Train the Model

        trainer.train()

        # Save the Model
        # saves the model, tokenizer, and training config
        trainer.save_model("./bert-imdb-model")  # Also saves tokenizer & config

        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            task="text-classification",
            artifact_path="bert_model")


        # Evaluate on the Test Set

        test_results = trainer.evaluate(test_dataset)
        print(f"Test Accuracy: {test_results['eval_accuracy']:.4f}")

        # Predict on the Test Set
        predictions = trainer.predict(test_dataset)

        # Confusion Matrix
        y_true = predictions.label_ids
        y_pred = predictions.predictions.argmax(-1)
        plt = plot_confusion_matrix(y_true, y_pred)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        # Delete the confusion matrix.png
        os.remove("confusion_matrix.png")



if __name__ == "__main__":
    main()