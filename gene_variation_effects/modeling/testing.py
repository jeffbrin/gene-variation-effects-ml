import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import numpy as np
from .utils import generate_confusion_matrix, calculate_f_score_from_conf_matrix, logits_to_prediction

from typing import Optional

def test_model(
        model: torch.nn.Module, 
        batch_size: int, 
        X: torch.Tensor, 
        labels: np.ndarray,
        embedding_features_columns: list[int],
        unique_gene_lists,
        positive_threshold: float = 0.5,
        criterion: Optional[torch.nn.modules.loss._Loss] = None) -> tuple[float, float]:
    """
    Tests the model, returning average loss and accuracy.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be tested
    batch_size : int
        The size of the batch to be given to the model
    X : torch.Tensor
        The input data
    labels : np.ndarray
        Labels for desired predictions
    embedding_features_columns : list[int]
        The column indices of features in the X tensor which use embedding
    positive_threshold : float
        The value above which a prediction will be considered positive, 0.5 by default.
    criterion : Optional[torch.nn.modules.loss._Loss], optional
        The criterion class used to evaluate loss. Defaults to BCEWithLogitsLoss if None is provided, by default None

    Returns
    -------
    tuple[float, float]
        Returns (average loss, accuracy) for the test data
    """

    model.eval()
    

    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    test_dataset = TensorDataset(X, labels)

    bincount = torch.bincount(labels.flatten().long())
    class_weights = 1 / bincount # 9:1 ratio in dataset
    sample_weights = class_weights[labels.flatten().long()]
    training_sampler = WeightedRandomSampler(sample_weights, len(labels), replacement=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=training_sampler)

    all_preditions = []
    all_targets = []
    total_f_score = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X, embedding_features_columns, unique_gene_lists)
            all_preditions.extend(predictions)
            all_targets.extend(batch_y)
            loss = criterion(predictions, batch_y)

            total_loss += loss.item() * batch_y.size(0)
            total_correct += (logits_to_prediction(predictions, positive_threshold) == batch_y).float().sum()
            total_samples += batch_y.size(0)

            confusion_matrix = generate_confusion_matrix(logits_to_prediction(predictions, positive_threshold), batch_y)
            f_score = calculate_f_score_from_conf_matrix(confusion_matrix)
            total_f_score += f_score * batch_y.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    avg_f_score = total_f_score / total_samples

    return avg_loss, accuracy, all_preditions, all_targets, avg_f_score
    

        