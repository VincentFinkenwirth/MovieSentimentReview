import optuna
import mlflow
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from tools.data_combination import load_data
from tools.viszualizer import  plot_confusion_matrix
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

import gc
import numpy as np
import pandas as pd
import os
import torch
from mlflow.models.signature import infer_signature

# GLOBAL
N_JOBS = 1 #Only 1 as GPU training
N_TRIALS = 20
#PATHS
PATH_DATA = "../data/preprocessed/split_data_v2"
MLFLOW_PATH = "../mlruns"

############################################################################################

# Function to clear GPU memory as I get CUDA out of memory errors
def clear_memory():
    """
    Clear GPU memory to avoid CUDA out-of-memory errors.

    1) Frees cached GPU memory (if available).
    2) Collects inter-process communication resources.
    3) Calls Python's garbage collector.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Release cached memory
        torch.cuda.ipc_collect()  # Collect inter-process communication resources
        gc.collect()


    # Use CatBoost gradient boosting model for finetuning
def build_model(trial):
    """
    Build and configure a CatBoostClassifier model for hyperparameter optimization.

    1) Suggests a range of hyperparameters from Optuna's search space.
    2) Initializes a CatBoostClassifier with those hyperparameters, using GPU.
    """
    # Test different hyperparameters
    depth = trial.suggest_int("depth", 4, 12)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
    iterations = trial.suggest_int("iterations", 50, 1000, log=True)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 20, log=True)
    border_count = trial.suggest_int("border_count", 32, 255)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1, step=0.1) # improves generalization
    random_strength = trial.suggest_float("random_strength", 0, 1, step=0.1) # adds noise/ randomness
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 50) # reduce complexity

    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        border_count=border_count,
        bagging_temperature=bagging_temperature,
        random_strength=random_strength,
        min_data_in_leaf=min_data_in_leaf,
        task_type="GPU",
        early_stopping_rounds=20,
        gpu_ram_part=0.7,
        max_ctr_complexity=1,
        verbose=False,
        random_seed=42)
    return model

def main():
    """
    Main function to run Optuna hyperparameter optimization and evaluate a
    CatBoost model on a final test set.

    1) Loads training, validation, and test data.
    2) Defines an Optuna objective function that trains and validates models.
    3) Uses MLflow to track all runs, hyperparameters, and metrics.
    4) Re-trains the best model on combined (train+val) data and evaluates on test.
    5) Logs final metrics and artifacts (such as a confusion matrix) to MLflow.
    """
    # Set MLflow
    mlflow.set_tracking_uri(MLFLOW_PATH)

    # Load data
    X_train, X_val, X_test, y_train,y_val, y_test = load_data(path=PATH_DATA, val_set=True, X_col="token")

    #create pool to use catboost with text data instead of classical vectorizer
    train_pool = Pool(X_train, y_train, text_features=[0])
    val_pool = Pool(X_val, y_val, text_features=[0])
    test_pool = Pool(X_test, y_test, text_features=[0])


    ####################### Nested objective function ########################################################
    def objective(trial):
        """
        Nested objective function for Optuna to train and validate the model.

        1) Clears GPU memory before each trial.
        2) Builds a CatBoost model with sampled hyperparameters.
        3) Trains the model on train_pool and evaluates on val_pool.
        4) Logs metrics and hyperparameters to MLflow.
        5) Returns the accuracy score as the primary optimization metric.
        """
        # Clear memory
        clear_memory()

        # Create pipeline
        model = build_model(trial)

        try:
            with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
                # Fit pipeline
                model.fit(train_pool, eval_set=val_pool)

                # Get predictions
                y_pred = model.predict(val_pool)

                # Compute metrics
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred)

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1", f1)

                # Log trial parameters
                mlflow.log_params(trial.params)

                # Return primary optimization metric
                return accuracy

        except Exception as e:  # Catch exceptions
            print(f"Trial failed: {str(e)}")
            raise optuna.TrialPruned()

        finally:  # Collect garbage
            del model
            clear_memory()


    ####################### Run optuna#########################################################

    # Set MLflow experiment
    mlflow.set_experiment(f"CatBoost_v2_{time.strftime('%Y-%m-%d-%H-%M-%S')}")

    with mlflow.start_run(run_name=f"Optuna_CatBoost_Finetuning_v2_{time.strftime('%Y-%m-%d-%H-%M-%S')}") as run:
        # Create pruner to stop unpromising trials early
        pruner = optuna.pruners.HyperbandPruner(min_resource=1,reduction_factor=3)
        # Create an Optuna study that tries to maximize accuracy
        study = optuna.create_study(direction="maximize",
                                    study_name=f"Optuna_CatBoost_Finetuning_v2_{time.strftime('%Y-%m-%d-%H-%M-%S')}",
                                    storage="sqlite:///optuna_CatBoost_CV.db",
                                    load_if_exists=False,
                                    pruner=pruner)

        # Optimize for a certain number of trials
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=N_JOBS)

        # Load best trial
        best_trial = study.best_trial
        # log best parameters and trial accuracy
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("best_trial_accuracy", best_trial.value)

        # Rebuild best model
        best_model= build_model(best_trial)

        # Combine train and val data for final model
        X_train_combined = pd.concat([X_train, X_val])
        y_train_combined = np.concatenate([y_train, y_val])
        # Create pool
        combined_pool = Pool(X_train_combined, y_train_combined, text_features=[0])
        # Fit best model on combined data
        best_model.fit(combined_pool)

        # Predict on test data
        y_pred_test = best_model.predict(test_pool)

        # Compute metrics
        accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Plot confusion matrix
        fig = plot_confusion_matrix(y_test, y_pred_test)
        fig.savefig("confusion_matrix_best_test.png")
        mlflow.log_artifact("confusion_matrix_best_test.png")
        os.remove("confusion_matrix_best_test.png")

        # Infere Signature
        signature = infer_signature(X_train_combined, best_model.predict(combined_pool))

        # Log model
        mlflow.catboost.log_model(best_model, "catboost_model", signature=signature)


if __name__ == "__main__":
    main()


