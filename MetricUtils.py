import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)

def display_training_history(history):
    """
    Display validation loss and accuracy curves.

    Parameters:
    -----------
    history : dict
        Dictionary containing 'val_loss' and 'val_acc'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot validation loss
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plot validation accuracy
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def confusionMatrix(y_true, y_pred, labels=None, cmap='viridis'):
    """
    Displays the confusion matrix for the model predictions.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth (true labels), one-hot encoded or class labels.
    y_pred : np.ndarray
        Predicted labels or probabilities.
    labels : list, optional
        List of label names for the confusion matrix.
    cmap : str, optional
        Colormap for the confusion matrix visualization.
    """
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels)
    disp.plot(cmap=cmap, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def evaluate_model(y_pred_proba, y_test, threshold=0.5):
    """
    Evaluate model performance on the test set.

    Parameters:
    -----------
    model : object
        A trained model with a `predict` method.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        True labels (one-hot encoded or class labels).
    threshold : float, optional
        Threshold for converting probabilities into class labels (default=0.5).
    multi_class : str, optional
        Multi-class ROC-AUC strategy ('ovr' or 'ovo').

    Returns:
    --------
    dict:
        Dictionary containing various performance metrics.
    """
    if len(y_pred_proba.shape) == 1 or y_pred_proba.shape[1] == 1:  # Binary classification
        y_pred = (y_pred_proba >= threshold).astype(int)
    else:  # Multi-class classification
        y_pred = np.argmax(y_pred_proba, axis=1)

    # Convert y_test to class labels if it's one-hot encoded
    if len(y_test.shape) > 1:
        y_test = np.argmax(y_test, axis=1)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    return metrics