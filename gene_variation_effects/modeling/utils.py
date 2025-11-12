import torch
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

def logits_to_prediction(logits: torch.Tensor, positive_threshold: float) -> torch.Tensor:
        return (torch.sigmoid(logits) > positive_threshold).type(torch.int)

# The proportion of predicted pathogenic which are correct
def calculate_precision_from_conf_matrix(matrix):
    return matrix['Predicted Pathogenic']['True Pathogenic'] / sum(matrix['Predicted Pathogenic'])
# Proportion of actual pathogenic variants which were predicted correctly
def calculate_recall_from_conf_matrix(matrix):
    return matrix['Predicted Pathogenic']['True Pathogenic'] / sum(matrix.loc['True Pathogenic'])
# Proportion of actual benign which were predicted correctly
def calculate_specificity_from_conf_matrix(matrix):
    return matrix['Predicted Benign']['True Benign'] / sum(matrix.loc['True Benign'])

def calculate_f_score_from_conf_matrix(matrix):
    precision = calculate_precision_from_conf_matrix(matrix)
    recall = calculate_recall_from_conf_matrix(matrix)
    return 2 * precision * recall / (precision + recall)

NEGATIVE_TRUTH = "True Benign"
POSITIVE_TRUTH = "True Pathogenic"
NEGATIVE_PREDICTION = "Predicted Benign"
POSITIVE_PREDICTION = "Predicted Pathogenic"
def generate_confusion_matrix(predictions, targets, normalize=None):
    cm_indices = [NEGATIVE_TRUTH, POSITIVE_TRUTH]
    cm_cols = [NEGATIVE_PREDICTION, POSITIVE_PREDICTION]
    confusion_matrix = metrics.confusion_matrix(targets, predictions, normalize=normalize)
    return pd.DataFrame(confusion_matrix, index = cm_indices, columns = cm_cols)

def get_matrix_labels():
     return NEGATIVE_TRUTH, POSITIVE_TRUTH, NEGATIVE_PREDICTION, POSITIVE_PREDICTION

def generate_roc_curve(prediction_logits, targets):
    AXIS_STEPS = 10
    points = []
    for threshold_pct in range(AXIS_STEPS):
        threshold = threshold_pct / AXIS_STEPS
        predictions = logits_to_prediction(torch.Tensor(prediction_logits), threshold)
        confusion_matrix = generate_confusion_matrix(predictions, targets, normalize=None)
        true_positive_rate = confusion_matrix[POSITIVE_PREDICTION][POSITIVE_TRUTH] / sum(confusion_matrix.loc[POSITIVE_TRUTH])
        false_positive_rate = confusion_matrix[POSITIVE_PREDICTION][NEGATIVE_TRUTH] / sum(confusion_matrix.loc[NEGATIVE_TRUTH])

        point = (false_positive_rate, true_positive_rate)
        points.append(point)
    return points

def plot_roc_curve(curve: list[tuple[int, int]], data_source: str, show_random=False):
    plt.plot([p[0] for p in curve], [p[1] for p in curve], label="Model")
    if show_random:
        plt.plot([0, 1], [0, 1], label="Random")
        plt.legend()
    plt.title(f"{data_source} ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

def show_confusion_matrix_heatmap(matrix, data_source):
    sn.heatmap(matrix, annot=True)
    plt.title(f"{data_source} Confusion Matrix (Row Normalized)")
    plt.show()
    plt.clf()