import torch
from sklearn import metrics
import pandas as pd

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