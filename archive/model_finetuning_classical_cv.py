

import optuna
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from tools.data_combination import load_data
import pandas as pd
from mlflow.models.signature import infer_signature
import os
import time
from tools.viszualizer import plot_confusion_matrix
from sklearn.model_selection import cross_validate
import numpy as np
import gc
from sklearn.model_selection import StratifiedKFold

############################################################################################################
## GLOBAL
N_JOBS = 6
N_TRIALS = 100
CROSS_VALIDATION = 3
# Models:
MODELS = ["random_forest"] # ["log_reg", "random_forest", "support_vector", "naive_bayes"]
# Vectorizers:
VECTORIZER = ["count", "tfidf"] # ["count", "tfidf"]

# PATHS
PATH_DATA = "../data/preprocessed/split_data_v1"
MLFLOW_PATH = "../mlruns"
MLFLOW_NAME = f"{"_".join(MODELS)}_v2"
# Timeout (max time for a trial)
TIMEOUT_MAX = 1200


###########################################################################################################



# Set mlflow tracking directory
mlflow.set_tracking_uri(MLFLOW_PATH)

############## Build the pipeline with different vectorizers and models ###############################
def build_pipeline(trial):
    # Choose vectorizer
    vectorizer_name = trial.suggest_categorical("vectorizer", VECTORIZER)

    if vectorizer_name == "count":
        # Test different hyperparameters - ranges already improved through previous runs
        ngram_str = trial.suggest_categorical("count__ngram_range", ["(1,1)", "(1,2)"])
        if ngram_str == "(1,1)":
            ngram_range = (1, 1)
        else:
            ngram_range = (1, 2)
        max_features = trial.suggest_int("count__max_features", 4000, 10000, log=True)

        vectorizer = CountVectorizer(ngram_range=ngram_range, analyzer="word", max_features=max_features)

    elif vectorizer_name == "tfidf":
        # Test different hyperparameters
        ngram_str = trial.suggest_categorical("tfidf__ngram_range", ["(1,1)", "(1,2)"])
        if ngram_str == "(1,1)":
            ngram_range = (1, 1)
        else:
            ngram_range = (1, 2)
        max_features = trial.suggest_int("tfidf__max_features", 4000, 10000, log=True)
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer="word", max_features=max_features)
    else:
        raise ValueError(f"Invalid vectorizer name: {vectorizer_name}")


    # Choose models
    classifier_name = trial.suggest_categorical("classifier", MODELS)

    if classifier_name == "log_reg":
        # Test different hyperparameters
        C = trial.suggest_float("log_reg__C", 1e-3, 1e1, log=True)
        penalty = trial.suggest_categorical("log_reg__penalty", ["l1", "l2"])
        max_iter = trial.suggest_int("log_reg__max_iter", 100, 1000)

        classifier = LogisticRegression(C=C, solver="saga", penalty=penalty, max_iter=max_iter, random_state=42)

    elif classifier_name == "random_forest":
        # Test different hyperparameters
        n_estimators = trial.suggest_int("random_forest__n_estimators", 100, 1000, log=True)
        max_depth = trial.suggest_int("random_forest__max_depth", 3, 50)
        min_samples_split = trial.suggest_int("random_forest__min_samples_split", 2, 20)
        criterion = trial.suggest_categorical("random_forest__criterion", ["gini", "entropy"])
        max_features = trial.suggest_categorical("random_forest__max_features", ["sqrt", "log2", None])

        classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                            min_samples_split=min_samples_split, criterion=criterion, max_features=max_features, random_state=42)

    elif classifier_name == "naive_bayes":
        # Test different hyperparameters
        alpha = trial.suggest_float("naive_bayes__alpha", 1e-3, 1e2, log=True)

        classifier = MultinomialNB(alpha=alpha)

    elif classifier_name == "support_vector":
        # Test different hyperparameters
        C = trial.suggest_float("support_vector__C", 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical("support_vector__kernel", ["linear", "rbf"])
        # Kernel-specific parameters
        if kernel == "rbf":
            gamma = trial.suggest_float("support_vector__gamma", 1e-5, 1, log=True)
        # Class weight for imbalanced datasets
        class_weight = trial.suggest_categorical("support_vector__class_weight", [None, "balanced"])
        # Define the classifier
        classifier = SVC(C=C,kernel=kernel,shrinking=True,class_weight=class_weight, probability=True,
            gamma=gamma if kernel=="rbf" else "scale",
            random_state=42, cache_size=1024)
    else: # Space for additional models
        raise ValueError(f"Invalid model name: {classifier_name}")

    # Build the pipeline
    pipeline = Pipeline([
        ("vect", vectorizer),
        ("clf", classifier)],
        memory="cache")

    return pipeline

############################ Run optuna and log using MLflow ############################################
def main():
    # Load data
    X_train, X_test, y_train, y_test = load_data(val_set=False, path=PATH_DATA, X_col="token")

############################# Nested objective function to avoid reloading data##########################
    def objective(trial):
        # Create pipeline
        pipeline = build_pipeline(trial)

        # Define scoring metrics
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score,zero_division=0),
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
                    n_jobs=1
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
            gc.collect()
########################################################################################################

    # Name experiment using date and time to distinguish between runs
    mlflow.set_experiment(f"{MLFLOW_NAME}_{time.strftime('%Y-%m-%d-%H-%M-%S')}")
    # Start a "parent" MLflow run to encompass all trials
    with mlflow.start_run(run_name="Optuna_Classical_Model_Finetuning") as run:
        # Create pruner to stop unpromising trials early
        pruner = optuna.pruners.HyperbandPruner(min_resource=1, reduction_factor=3)

        # Create an Optuna study that tries to maximize accuracy
        study = optuna.create_study(direction="maximize",
                                    study_name=f"Optuna_Classical_Model_Finetuning_{time.strftime('%Y-%m-%d-%H-%M-%S')}",
                                    storage="sqlite:///optuna_classical_CV.db",
                                    load_if_exists=False,
                                    pruner=pruner)

        # Optimize for a certain number of trials
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=N_JOBS)

        # Retrieve the best trial
        best_trial = study.best_trial
        print("Best Trial ID:", best_trial.number)
        print("Best Trial Value (Accuracy):", best_trial.value)

        # Log the best trial info in the parent run
        mlflow.log_param("best_trial_id", best_trial.number)
        mlflow.log_metric("best_val_accuracy", best_trial.value)
        mlflow.log_params(best_trial.params)

        # reconstruct the best pipeline using best_trial.params
        best_pipeline = build_pipeline(best_trial)
        best_pipeline.fit(X_train, y_train)

        # Evaluate on test set
        y_pred_best = best_pipeline.predict(X_test)
        final_acc = accuracy_score(y_test, y_pred_best)
        mlflow.log_metric("final_test_accuracy", final_acc)

        # Log the signature of the model
        signature = infer_signature(X_train, best_pipeline.predict(X_train))
        mlflow.sklearn.log_model(best_pipeline, signature=signature, artifact_path="best_pipeline")

        # Confusion matrix
        # Create a directory to save confusion matrix IF it does not exist
        os.makedirs("confusion_matrix", exist_ok=True)
        #  Create using premade function
        fig = plot_confusion_matrix(y_test, y_pred_best)
        # Save the confusion matrix
        fig.savefig(f"confusion_matrix/confusion_matrix_best_test.png")
        # Log the confusion matrix as an artifact
        mlflow.log_artifact(f"confusion_matrix/confusion_matrix_best_test.png")

        # Log parameters of the best pipeline
        mlflow.log_params(best_trial.params)

    # Print ending confirmation
    print("Finished!")


if __name__ == "__main__":
    main()