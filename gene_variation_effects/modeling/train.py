from loguru import logger
import torch
import numpy as np

from typing import Optional

def run_training_loop(
        model: torch.nn.Module, 
        epochs: int, 
        training_X: torch.Tensor, 
        validation_X: torch.Tensor, 
        training_labels: np.ndarray, 
        validation_labels: np.ndarray,
        embedding_features_columns: list[int],
        criterion: Optional[torch.optim.Optimizer] = None,
        optimizer: Optional[torch.nn.modules.loss._Loss] = None) -> ...:

    def get_current_epoch_data(data: torch.Tensor, epoch: int, epochs: int) -> torch.Tensor:
        return data[
            (data.shape[0] // epochs) * epoch:
            (data.shape[0] // epochs) * (epoch+1)]

    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        epoch_input = get_current_epoch_data(training_X, epoch, epochs)
        epoch_training_labels = get_current_epoch_data(training_labels, epoch, epochs)
        epoch_valdiation = get_current_epoch_data(validation_X, epoch, epochs)
        epoch_validation_labels = get_current_epoch_data(validation_labels, epoch, epochs)


        output = model(epoch_input, embedding_features_columns)
        # logits = model(x, embedding_features_columns)
        # logger.info(torch.isnan(output).any(), torch.isinf(output).any())
        loss = criterion(output, epoch_training_labels)
        loss.backward()
        optimizer.step()

        logger.info(f"Epoch {epoch}: loss={loss.item():.4f}")

        