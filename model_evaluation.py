import pandas as pd
import os
import mlflow
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tools.viszualizer import plot_confusion_matrix, plot_roc_curve
import joblib as jb
from catboost import Pool
import numpy as np
from models.model_finetuning_transformer import preprocess_function

# GLOBAL
DATA_PATH = "data/test_datasets/rotten_test.csv"
# Save path
SAVE_PATH = "data/test_datasets/results_v2/"

# Model paths v2
BERT = 'runs:/830d573240b84785bfc8360b0d074a13/bert_model'
SVM = 'runs:/6ae2878823714c789e4bffa97a621683/best_pipeline'
CATBOOST = 'runs:/7d34cc89d2b4437da68b008c9dd982a4/catboost_model'
RANDOM_FOREST = 'runs:/4e0f4cddc19c4094ad6648d0bc0d959c/best_pipeline'
LOG_REG = 'runs:/84e6b11d8ceb48ff9168f1fd3ae6189d/best_pipeline'
NAIVE_BAYES = 'runs:/4564d18622a64c7bbdeaba61929c8db5/best_pipeline'

# Model paths v1


# Dictionary of model names
MODEL_NAMES = {
    "rf": "Random Forest",
    "nb": "Naive Bayes",
    "svm": "Support Vector Machine",
    "log_reg": "Logistic Regression",
    "catboost": "CatBoost",
    "bert": "BERT"
}
# Add probability columns to data for classical models
def add_probability_column(model, data, model_name, text_column="token"):
    ''' Add probability column to data based on model type'''
    try:
        if hasattr(model, "predict_proba"):
            # predict_proba returns an array of shape (n_samples, n_classes)
            proba = model.predict_proba(data[text_column])
            # Assuming positive class is at index 1
            data[f"{model_name}_proba"] = proba[:, 1]
        elif hasattr(model, "decision_function"):
            # Use decision_function if probabilities are not available
            data[f"{model_name}_proba"] = model.decision_function(data[text_column])
        else:
            print(f"Model '{model_name}' does not support probability predictions.")
    except Exception as e:
        print(f"Error computing prob for '{model_name}': {e}")
    return data

# Predict using classical models
def predict_classical(data, RANDOM_FOREST=RANDOM_FOREST, NAIVE_BAYES=NAIVE_BAYES, SVM=SVM, LOG_REG=LOG_REG):
    ''' Apply predictions and probabilities to data using classical models '''
    # Load models (skleanr pipelines including vectorizer)
    rf = mlflow.sklearn.load_model(RANDOM_FOREST)
    nb = mlflow.sklearn.load_model(NAIVE_BAYES)
    svm = mlflow.sklearn.load_model(SVM)
    log_reg = mlflow.sklearn.load_model(LOG_REG)

    # Predict and add probabilities if available
    data["rf_pred"] = rf.predict(data["token"])
    data = add_probability_column(rf, data, "rf", "token")
    print("Random Forest done")

    data["nb_pred"] = nb.predict(data["token"])
    data = add_probability_column(nb, data, "nb", "token")
    print("Naive Bayes done")

    data["svm_pred"] = svm.predict(data["token"])
    data = add_probability_column(svm, data, "svm", "token")
    print("SVM done")

    data["log_reg_pred"] = log_reg.predict(data["token"])
    data = add_probability_column(log_reg, data, "log_reg", "token")
    print("Logistic Regression done")

    return data

# Predict using CatBoost
def predict_catboost(data, CATBOOST=CATBOOST):
    ''' Apply predictions and probabilities to data using CatBoost '''
    # Load model
    catboost = mlflow.catboost.load_model(CATBOOST)

    # Turn data into pool
    data_pool = Pool(data["token"], text_features=[0])

    # Predict
    data["catboost_pred"] = catboost.predict(data_pool)
    # Try to predict probabilities
    try:
        proba = catboost.predict(data_pool, prediction_type='Probability')
        # If proba has two columns, use the second column (positive class)
        if proba.ndim == 2 and proba.shape[1] > 1:
            data["catboost_proba"] = proba[:, 1]
        else:
            data["catboost_proba"] = proba
    except Exception as e:
        print(f"Could not compute probabilities for CatBoost: {e}")

    print("CatBoost done")

    return data

# Predict using BERT
def predict_bert(data, BERT=BERT):
    ''' Apply predictions and probabilities to data using BERT '''
    # Load model
    bert = mlflow.transformers.load_model(BERT)

    # Format data
    processed = preprocess_function(data)
    text = processed["text"]

    # Predict
    predictions = bert.predict(text)
    # Extract sentiment from predictions
    data["bert_pred"] = [1 if x["label"] == "LABEL_1" else 0 for x in predictions]
    # Extract confidence from predictions for AUC-ROC
    data["bert_proba"] = [
        pred["score"] if pred["label"] == "LABEL_1" else 1 - pred["score"]
        for pred in predictions]
    print("BERT done")

    return data

def appy_predictions(data_path):
    ''' Apply predictions of all models to data and save to disk '''
    # Load data
    data = pd.read_csv(data_path)

    # Predict classical models
    data = predict_classical(data)

    # Predict catboost
    data = predict_catboost(data)

    # Predict BERT
    data = predict_bert(data)

    # Save data
    data.to_csv(f"{SAVE_PATH}predictions.csv", index=False)
    return data


def generate_classification_report_csv(y_true, y_pred, model_name, save_path):
    ''' Generate classification report and save to CSV. Input: y_true, y_pred, model_name, save_path '''
    #  Create report
    report_dict = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0  # handles division by zero
    )

    # Convert the dictionary to a DataFrame
    report_df = pd.DataFrame(report_dict).transpose()

    # Save to CSV
    report_df.to_csv(f"{save_path}report/{model_name}_classification_report.csv", index=True)

def evaluate_predictions(data_path="predictions.csv", data=None, save_path=SAVE_PATH, plot_confusion_matrix=True, plot_roc_curve=True):
    ''' Evaluate predictions and save metrics to disk. If data is None, load from data_path. If plot_confusion_matrix is True or plot_roc_curve is True, directory in save_path is requiered. '''
    if data is None:
        data = pd.read_csv(data_path)

    # temporary storage
    final = []

    #Loop through models
    for model in ["rf", "nb", "svm", "log_reg", "catboost", "bert"]:
        # Compute metrics
        accuracy = accuracy_score(data["label"], data[f"{model}_pred"])
        precision = precision_score(data["label"], data[f"{model}_pred"])
        recall = recall_score(data["label"], data[f"{model}_pred"])
        f1 = f1_score(data["label"], data[f"{model}_pred"])
        # Add to final
        final.append(([model, accuracy, f1, precision, recall]))

        # Generate classification report
        generate_classification_report_csv(data["label"], data[f"{model}_pred"], model_name=model, save_path=save_path)

        # Plot confusion matrix and save to confusion matrix folder
        if plot_confusion_matrix:
            cm = plot_confusion_matrix(data["label"], data[f"{model}_pred"], model_name=MODEL_NAMES[model], include_percentages=True)
            cm.savefig(f"{save_path}confusion_matrix/{model}_confusion_matrix.png")

        # Plot ROC curve and save to roc curve folder
        if plot_roc_curve:
            roc = plot_roc_curve(data["label"], data[f"{model}_proba"], model_name=MODEL_NAMES[model])
            roc.savefig(f"{save_path}roc/{model}_roc_curve.png")

    metrics_df = pd.DataFrame(final, columns=["Model", "Accuracy", "F1", "Precision", "Recall"],)
    metrics_df.to_csv(f"{save_path}metrics.csv", index=True)
    print(metrics_df)


if __name__ == "__main__":
    data = appy_predictions(DATA_PATH)
    evaluate_predictions(data=data, save_path=SAVE_PATH)
