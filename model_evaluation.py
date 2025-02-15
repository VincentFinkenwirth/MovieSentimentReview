import pandas as pd
import mlflow
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from tools.viszualizer import plot_confusion_matrix, plot_roc_curve
from catboost import Pool
from models.model_finetuning_transformer import preprocess_function

# GLOBAL
DATA_PATH = "data/test_datasets/rotten_test.csv"
# Save path of metrics and plots
SAVE_PATH = "evaluation_results/results_v2/"
# Save path of predictions
SAVE_PREDICTIONS = "data/predictions//predictions_v2.csv"

# Model paths v2
BERT = 'runs:/830d573240b84785bfc8360b0d074a13/bert_model'
SVM = 'runs:/6ae2878823714c789e4bffa97a621683/best_pipeline'
CATBOOST = 'runs:/1aa7d53b20e7470998d26ca603bfccc3/catboost_model'
RANDOM_FOREST = 'runs:/4e0f4cddc19c4094ad6648d0bc0d959c/best_pipeline'
LOG_REG = 'runs:/84e6b11d8ceb48ff9168f1fd3ae6189d/best_pipeline'
NAIVE_BAYES = 'runs:/4564d18622a64c7bbdeaba61929c8db5/best_pipeline'


# Dictionary of full model names
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
    """
    Add a model-specific probability or decision-function output column to 'data' used for classical models.

    1) If the model has a predict_proba method, it uses the second column for the
       positive-class probability.
    2) If the model has a decision_function method, it stores those values instead.
    3) Handles exceptions if the model does not support probability predictions.

    Parameters:
        model: A trained model (usually a scikit-learn pipeline) to make probability predictions.
        data (pd.DataFrame): The dataframe where a new probability column should be added.
        model_name (str): Short name or key used to label the new column.
        text_column (str): Column in 'data' containing the text or features the model consumes.

    Returns:
        pd.DataFrame: The updated dataframe with an added probability column (if applicable).
    """
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
def predict_classical(data):
    """
    Predict labels and add probabilities for multiple classical models (RF, NB, SVM, LogReg).

    1) Loads models from MLflow URIs.
    2) Predicts labels on the 'token' column.
    3) Adds probability columns if supported.

    Returns:
        pd.DataFrame: The dataframe augmented with predictions and probabilities for each model.
    """
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
def predict_catboost(data, datatype="text"):
    """
    Apply predictions and probabilities to data with a CatBoost model.

    1) Loads the CatBoost model from MLflow.
    2) Builds a CatBoost 'Pool' from the 'token' column.
    3) Predicts class labels.
    4) Tries to predict probabilities using 'predict(..., prediction_type="Probability")'.

    Parameters:
        data (pd.DataFrame): Dataset to predict on.
        datatype (str): Column name in 'data' containing the text data. Default is 'text', option to use 'token'.

    Returns:
        pd.DataFrame: Dataframe augmented with CatBoost predictions and probability columns.
    """
    # Load model
    catboost = mlflow.catboost.load_model(CATBOOST)

    # Turn data into pool
    data_pool = Pool(data[datatype], text_features=[0])

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
def predict_bert(data):
    """
    Apply predictions and probabilities to data using BERT model.

    1) Loads a BERT pipeline from MLflow Transformers.
    2) Preprocesses the data using custom 'preprocess_function'.
    3) Predicts labels (0 or 1) and extracts probabilities for each row.

    Parameters:
        data (pd.DataFrame): Dataset containing a 'text' column or one suitable for transformation.

    Returns:
        pd.DataFrame: Dataframe augmented with BERT predictions and probability columns.
    """
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
    """
    Apply all model predictions (classical + CatBoost + BERT) to a dataset,
    and save the augmented data with predictions to a CSV file.

    1) Reads the CSV from 'data_path'.
    2) Passes the data through 'predict_classical', 'predict_catboost', and 'predict_bert'.
    3) Saves the resulting dataframe with new columns for each model's predictions and probabilities/confidence.

    Parameters:
        data_path (str): File path to the CSV file containing at least 'token' and 'text' columns.

    Returns:
        pd.DataFrame: Dataframe with appended prediction and probability columns from all models.
    """
    # Load data
    data = pd.read_csv(data_path)

    # Predict classical models
    data = predict_classical(data)

    # Predict catboost
    data = predict_catboost(data)

    # Predict BERT
    data = predict_bert(data)

    # Save data
    data.to_csv(SAVE_PREDICTIONS, index=False)
    return data


def generate_classification_report_csv(y_true, y_pred, model_name, save_path):
    """
    Generate and save a classification report (precision, recall, f1, etc.) to CSV.

    Parameters:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels from the model.
        model_name (str): Short name or key for the model, used in output filename.
        save_path (str): Directory in which the CSV report will be saved.
    """
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

def evaluate_predictions(data=None, data_path="predictions.csv", save_path=SAVE_PATH, plot_cm=True, plot_roc=True):
    """
    Evaluate model predictions and save metrics (accuracy, F1, precision, recall),
    as well as artifacts (confusion matrices, ROC curves, classification reports).

    1) Either use the provided 'data' dataframe or load from 'data_path'.
    2) Iterates over each model in MODEL_NAMES to compute metrics.
    3) Saves metrics, confusion matrices, ROC curves, and classification reports.

    Parameters:
        data (pd.DataFrame, optional): Dataframe with true and predicted labels. If None, loads CSV.
        data_path (str): Path to the CSV file if 'data' is not provided.
        save_path (str): Folder to which all evaluation files are saved.
        plot_cm (bool): Whether to create and save confusion matrices.
        plot_roc (bool): Whether to create and save ROC curves.
    """
    if data is None:
        data = pd.read_csv(data_path)

    # temporary storage
    final = []

    # Loop through models
    for model in ["rf", "nb", "svm", "log_reg", "catboost", "bert"]:
        # Compute metrics
        accuracy = accuracy_score(data["label"], data[f"{model}_pred"])
        precision = precision_score(data["label"], data[f"{model}_pred"], average="macro")
        recall = recall_score(data["label"], data[f"{model}_pred"], average="macro")
        f1 = f1_score(data["label"], data[f"{model}_pred"], average="macro")
        # Add to final
        final.append(([model, accuracy, f1, precision, recall]))

        # Generate classification report
        generate_classification_report_csv(data["label"], data[f"{model}_pred"], model_name=model, save_path=save_path)

        # Plot confusion matrix and save to confusion matrix folder
        if plot_cm:
            cm = plot_confusion_matrix(data["label"], data[f"{model}_pred"],
                                       model_name=MODEL_NAMES[model], include_percentages=True)
            cm.savefig(f"{save_path}confusion_matrix/{model}_confusion_matrix.png")

        # Plot ROC curve and save to roc curve folder
        if plot_roc:
            roc = plot_roc_curve(data["label"], data[f"{model}_proba"], model_name=MODEL_NAMES[model])
            roc.savefig(f"{save_path}roc/{model}_roc_curve.png")

    metrics_df = pd.DataFrame(final, columns=["Model", "Accuracy", "F1", "Precision", "Recall"],)
    metrics_df.to_csv(f"{save_path}metrics.csv", index=False)
    print(metrics_df)


if __name__ == "__main__":
    # Apply predictions to test data
    #appy_predictions(DATA_PATH)
    # Evaluate predictions
    data = pd.read_csv("data/predictions/predictions_v2.csv")
    evaluate_predictions(data=data, save_path=SAVE_PATH, plot_cm=True, plot_roc=True)
