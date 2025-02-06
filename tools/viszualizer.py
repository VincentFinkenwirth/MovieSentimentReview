# TODO
# Create confusion matrix function
# Create ROC curve function

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

# Uniform confusion matrix plot accross different models
def plot_confusion_matrix(y_true, y_pred, model_name=None, class_names=None, include_percentages=False):
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
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    return plt

