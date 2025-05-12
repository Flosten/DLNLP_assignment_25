"""
This module contains functions for visualizing data and model performance.

The functions include:
- `plot_hist`: Plots a histogram of the tokens.
- `plot_learning_curve`: Plots the learning curve of the model.
- `evalute_the_preformance`: Evaluates the model performance using accuracy and confusion matrix.

"""

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


def plot_hist(data):
    """
    Plot a histogram of the data.

    Args:
        data (list): The data to plot.

    Returns:
        fig, ax: The figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
    # ax.set_title("Histogram of Data")
    ax.set_xlabel("Token number")
    ax.set_ylabel("sentences")
    return fig, ax


def plot_learning_curve(train_loss, val_loss):
    """
    Plot the learning curve of the model.

    Args:
        train_loss (list): The training loss values.
        val_loss (list): The validation loss values.

    Returns:
        fig, ax: The figure and axis objects.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Training Loss", color="blue")
    ax.plot(epochs, val_loss, label="Validation Loss", color="orange")
    ax.set_xticks(epochs)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()

    return fig, ax


def evalute_the_preformance(pred, true):
    """
    Evaluate the performance of the model using accuracy and confusion matrix.

    Args:
        pred (list): The predicted labels.
        true (list): The true labels.

    Returns:
        acc (float): The accuracy of the model.
        fig, ax: The figure and axis objects for the confusion matrix.
    """
    # Calculate accuracy
    acc = accuracy_score(true, pred)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(10, 6))
    cm = confusion_matrix(true, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues")
    # ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    xticks = ["rotten", "fresh"]
    yticks = ["rotten", "fresh"]
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)

    return acc, fig, ax


def attention_score_vis(attention_score, masks):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(4):
        token_len = masks[i].sum()
        attention_score_i = attention_score[i][:token_len].squeeze()
        # x_ticks = range(1, token_len + 1)
        axes[i].bar(
            range(len(attention_score_i)), attention_score_i, color="skyblue", alpha=0.8
        )
        axes[i].set_xlabel("Token position")
        axes[i].set_ylabel("Attention score")

        subtitles = [
            "(a) Movie Review 1",
            "(b) Movie Review 2",
            "(c) Movie Review 3",
            "(d) Movie Review 4",
        ]

        if subtitles:
            axes[i].text(
                0.5,
                -0.15,
                subtitles[i],
                transform=axes[i].transAxes,
                ha="center",
                va="top",
                fontsize=11,
            )
    fig.subplots_adjust(hspace=0.3)
    # plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    return fig, axes
