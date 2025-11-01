from loguru import logger
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from typing import Optional, Any

def run_training_loop(
        model: torch.nn.Module, 
        batch_size: int, 
        training_X: torch.Tensor, 
        validation_X: torch.Tensor, 
        training_labels: np.ndarray, 
        validation_labels: np.ndarray,
        embedding_features_columns: list[int],
        patience: int = 5,
        criterion: Optional[torch.nn.modules.loss._Loss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None
        ) -> tuple[dict[str, Any], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Trains the given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained
    batch_size : int
        The number of predictions per epoch
    training_X : torch.Tensor
        The training input data
    validation_X : torch.Tensor
        The validation input data
    training_labels : np.ndarray
        The correct predictions for the given training_X input data. Must be in shape (X, 1), where X is the number of rows in the input.
    validation_labels : np.ndarray
        The correct predictions for the given validation_X input data. Must be in shape (X, 1), where X is the number of rows in the input.
    embedding_features_columns : list[int]
        The columns of the input data which use embedded encoding
    patience : int, optional
        The number of epochs which can pass without improving on the best validation loss before ending the training, by default 5
    criterion : Optional[torch.nn.modules.loss._Loss], optional
        The loss function, by default None
    optimizer : Optional[torch.optim.Optimizer], optional
        The desired optimizer (Ex. Adam), by default None

    Returns
    -------
    tuple[dict[str, Any], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]
        A dictionary representation of the model parameters with the best validation loss, training losses by epoch, validation losses by epoch, training accuracy by epoch, and validation accuracy by epoch.
    """
    
    model.train()

    if criterion is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_losses = []
    validation_losses = []
    training_accuracies = []
    val_accuracies = []
    best_validation_loss = float('inf')
    sequential_worse_valiations = 0
    
    training_data = TensorDataset(training_X, training_labels)
    validation_data = TensorDataset(validation_X, validation_labels)

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_data, batch_size=batch_size * 2)

    for epoch, ((epoch_training_X, epoch_training_y), (epoch_val_X, epoch_val_y)) in enumerate(zip(train_loader, val_loader)):
        training_predictions = model(epoch_training_X, embedding_features_columns)
        loss = criterion(training_predictions, epoch_training_y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            validation_predictions = model(epoch_val_X, embedding_features_columns)
            validation_loss = criterion(validation_predictions, epoch_val_y)

        training_losses.append(loss.detach())
        validation_losses.append(validation_loss.detach())

        epoch_accuracy = ((training_predictions >= 0.5).type(torch.int) == epoch_training_y).float().mean()
        epoch_val_accuracy = ((validation_predictions >= 0.5).type(torch.int) == epoch_val_y).float().mean()

        training_accuracies.append(epoch_accuracy.detach())
        val_accuracies.append(epoch_val_accuracy.detach())

        logger.info(f"Epoch {epoch+1}: accuracy={epoch_accuracy.item():.4f}")
        logger.info(f"Epoch {epoch+1}: validation accuracy={epoch_val_accuracy.item():.4f}")
        logger.info(f"Epoch {epoch+1}: validation loss={validation_loss.item():.4f}")

        # Kill training if no longer improving validation loss
        if validation_loss >= best_validation_loss:
            sequential_worse_valiations += 1
        else:
            best_validation_loss = validation_loss
            sequential_worse_valiations = 0
            optimal_model_dict = model.state_dict()


        if sequential_worse_valiations >= patience:
            logger.info(f"Validation loss has not improved in {patience} epochs. Prematurely ending training at epoch {epoch+1}")
            break

    return optimal_model_dict, training_losses, validation_losses, training_accuracies, val_accuracies
    

        