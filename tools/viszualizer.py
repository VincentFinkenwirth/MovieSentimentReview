# Description: This file contains the functions to visualize the confusion matrix and ROC curve for the models.
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


# Uniform confusion matrix plot accross different models
def plot_confusion_matrix(y_true, y_pred, model_name=None, class_names=None, include_percentages=False):
    """
    Plot a confusion matrix.

    1) Computes a confusion matrix using sklearn.
    2) Optionally includes percentage annotations within each cell.
    3) Customizes the heatmap with labels, a title (including an optional model name),
       and a clear layout.

    Parameters:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        model_name (str, optional): Name of the model for labeling in the plot title.
        class_names (list, optional): Names for the two classes. Defaults to
                                      ["Negative Review", "Positive Review"].
        include_percentages (bool, optional): Whether to show percentages in the
                                             confusion matrix cells.

    Returns:
        matplotlib.figure.Figure: The figure object containing the confusion matrix plot.
    """
    # Default class names if none provided
    if class_names is None:
        class_names = ["Negative Review", "Positive Review"]

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Prepare custom annotations
    if include_percentages:
        total = np.sum(cm)
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                # Compute percentage relative to the total count
                perc = (cm[i, j] / total * 100) if total > 0 else 0
                annot[i, j] = f"{cm[i, j]}\n({perc:.2f}%)"
    else:
        annot = cm

    # Create the figure and axis with an appropriate size
    fig, ax = plt.subplots(figsize=(7, 6))

    # Plot the heatmap with a visually appealing colormap and gridlines for clarity
    sns.heatmap(cm, annot=annot, fmt="",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar=False, linewidths=0.5, linecolor='gray')

    # Set axis labels with extra padding and adjust font sizes
    ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=10)
    ax.set_ylabel("True Labels", fontsize=14, labelpad=10)

    # Create a title that includes the model name, if provided
    title = "Confusion Matrix"
    if model_name:
        title += f" - {model_name}"
    ax.set_title(title, fontsize=16, pad=15)

    # Adjust layout to prevent overlapping elements
    plt.tight_layout()

    return fig


def plot_roc_curve(y_true, y_proba, model_name):
    """
       Plot the ROC curve for classifier's probability predictions.

       1) Uses roc_curve and roc_auc_score from sklearn to compute the FPR and TPR.
       2) Plots the ROC curve with AUC annotation in the legend.
       3) Includes a diagonal line representing random chance.

       Parameters:
           y_true (array-like): True binary class labels.
           y_proba (array-like): Predicted probabilities for the positive class.
           model_name (str): Name of the model for labeling the plot title.

       Returns:
           matplotlib.pyplot: The pyplot module with the ROC curve plot.
       """
    # Calculate false positive rates, true positive rates, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # Compute the area under the curve (AUC)
    auc_score = roc_auc_score(y_true, y_proba)

    # Create plot
    plt.figure(figsize=(6, 5))
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
    # Plot a random-chance baseline
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")

    # Set plot labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    return plt


def plot_summary_roc(data, models=None):
    """
    Plot the ROC curve for multiple models on the same plot.

    1) Extracts the true labels and predicted probabilities for each model.
    2) Calls plot_roc_curve for each model and plots them together.
    3) Includes a diagonal line representing random chance.

    Parameters:
        data (dict): A dictionary containing the true labels and predicted probabilities
                     for each model.
        models (list, optional): A list of model to include. Defaults to
                                    ["rf", "log_reg", "nb", "catboost", "bert", "svm"].

    Returns:
        matplotlib.pyplot: The pyplot module with the ROC curve plot.
    """
    # Default models if none provided
    if models is None:
        models = ["rf", "log_reg", "nb", "catboost", "bert", "svm"]
    # Get true values
    y_true = data["label"]
    # Create the plot
    plt.figure(figsize=(8, 7))
    # Plot the ROC curve for each model
    for model_name in models:
        # Get the probabilities
        y_proba = data[f"{model_name}_proba"]
        # Plot the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc_score = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
    # Plot a random-chance baseline
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    # Set plot labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return plt
