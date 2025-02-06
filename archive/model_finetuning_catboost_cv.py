import optuna
import mlflow
from catboost import CatBoostClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from tools.data_combination import load_data
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mlflow.catboost import log_model
from tools.viszualizer import  plot_confusion_matrix
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
import gc
import numpy as np
import os
import torch
from mlflow.models.signature import infer_signature

#############################################################################################
# GLOBAL
N_JOBS = 1
N_TRIALS = 20
CROSS_VALIDATION = 3
TASK_TYPE = "GPU"  # Set to "CPU" if you don't have a GPU
#PATHS
PATH_DATA = "../data/preprocessed/split_data_v1"
MLFLOW_PATH = "../mlruns"

############################################################################################

# Function to clear GPU memory as I get CUDA out of memory errors
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory
        torch.cuda.ipc_collect()  # Collect inter-process communication resources
        gc.collect()


    # Use CatBoost gradient boosting model for finetuning
def build_pipeline(trial):
    # Choose vectorizer
    vectorizer_name = trial.suggest_categorical("vectorizer", ["count", "tfidf"])

    if vectorizer_name == "count":
        # Test different hyperparameters
        ngram_str = trial.suggest_categorical("count__ngram_range", ["(1,1)", "(1,2)"])
        if ngram_str == "(1,1)":
            ngram_range = (1, 1)
        else:
            ngram_range = (1, 2)
        max_features = trial.suggest_int("count__max_features", 4000, 8000, log=True)

        vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer="word", max_features=max_features)

    elif vectorizer_name == "tfidf":
        # Test different hyperparameters
        ngram_str = trial.suggest_categorical("tfidf__ngram_range", ["(1,1)", "(1,2)"])
        if ngram_str == "(1,1)":
            ngram_range = (1, 1)
        else:
            ngram_range = (1, 2)
        max_features = trial.suggest_int("tfidf__max_features", 4000, 8000, log=True)
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer="word", max_features=max_features)


    # Test different hyperparameters
    depth = trial.suggest_int("depth", 4, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.3, log=True)
    iterations = trial.suggest_int("iterations", 50, 400, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10, log=True)

    model = CatBoostClassifier(depth=depth,
                               learning_rate=learning_rate,
                               iterations=iterations,
                               l2_leaf_reg=l2_leaf_reg,
                               task_type=TASK_TYPE,
                               early_stopping_rounds=20,
                               gpu_ram_part=0.5,
                               used_ram_limit="2GB",
                               max_ctr_complexity=1,
                               verbose=False)

    # Create pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("model", model.copy())],
         memory="cache")

    return pipeline

def main():
    # Set MLflow
    mlflow.set_tracking_uri(MLFLOW_PATH)

    # Load data
    X_train, X_test, y_train, y_test = load_data(path=PATH_DATA, val_set=False, X_col="token")

    ####################### Nested objective function ########################################################
    def objective(trial):
        # Clear memory
        clear_memory()
        # Load data

        # Create pipeline
        pipeline = build_pipeline(trial)

        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }

        # Perform cross-validation
        cv = StratifiedKFold(n_splits=CROSS_VALIDATION, shuffle=True, random_state=42)

        try:
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                # Perform cross-validation
                cv_results = cross_validate(
                    pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    pre_dispatch=1 # Avoid CUDA out of memory errors by limiting parallelism
                )

                # Log metrics
                for metric, scores in cv_results.items():
                    if metric.startswith('test_'):
                        mlflow.log_metric(f"mean_{metric[5:]}", scores.mean())
                        mlflow.log_metric(f"std_{metric[5:]}", scores.std())

                # Log trial parameters
                mlflow.log_params(trial.params)

                # Return primary optimization metric
                return cv_results['test_accuracy'].mean()

        except Exception as e:  # Catch exceptions
            print(f"Trial failed: {str(e)}")
            raise optuna.TrialPruned()

        finally:  # Collect garbage
            clear_memory()

    ####################### Run optuna#########################################################
    # Set MLflow experiment
    mlflow.set_experiment(f"Optuna_CatBoost_Finetuning_CV_{time.strftime('%Y-%m-%d-%H-%M-%S')}_v2")
    with mlflow.start_run(run_name="Optuna_CatBoost_Finetuning") as run:
        # Create pruner to stop unpromising trials early
        pruner = optuna.pruners.HyperbandPruner(min_resource=1,reduction_factor=3)
        # Create an Optuna study that tries to maximize accuracy
        study = optuna.create_study(direction="maximize",
                                    study_name="Optuna_CatBoost_Finetuning_CV",
                                    storage="sqlite:///optuna_CatBoost_CV.db",
                                    load_if_exists=True,
                                    pruner=pruner)

        # Optimize for a certain number of trials
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=N_JOBS)

        # Test best model
        best_trial = study.best_trial
        pipeline = build_pipeline(best_trial)
        X_train, X_test, y_train, y_test = load_data(return_string=True, path=PATH_DATA, val_set=False)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Plot confusion matrix
        fig = plot_confusion_matrix(y_test, y_pred)
        fig.savefig(f"confusion_matrix/confusion_matrix_best_test.png")
        mlflow.log_artifact(f"confusion_matrix/confusion_matrix_best_test.png")

        # Infere Signature
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, "best_model", signature=signature)
        # Log best trial parameters
        mlflow.log_params(best_trial.params)




if __name__ == "__main__":
    main()


