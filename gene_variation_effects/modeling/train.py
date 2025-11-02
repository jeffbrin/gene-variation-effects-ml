from loguru import logger
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

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
        positive_threshold: float = 0.5,
        positive_class_loss_weight: float = None,
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
    
    def logits_to_prediction(logits: torch.Tensor) -> torch.Tensor:
        return (torch.sigmoid(logits) >= positive_threshold).type(torch.int)

    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    training_losses = []
    validation_losses = []
    training_accuracies = []
    val_accuracies = []
    best_validation_loss = float('inf')
    sequential_worse_valiations = 0
    
    training_data = TensorDataset(training_X, training_labels)
    validation_data = TensorDataset(validation_X, validation_labels)

    bincount = torch.bincount(training_labels.flatten().long())
    class_weights = 1 / bincount # 9:1 ratio in dataset
    sample_weights_training = class_weights[training_labels.flatten().long()]
    training_sampler = WeightedRandomSampler(sample_weights_training, len(training_labels), replacement=True)
    # training_sampler = None
    sample_weights_val = class_weights[validation_labels.flatten().long()]
    val_sampler = WeightedRandomSampler(sample_weights_val, len(validation_labels), replacement=True)
    # val_sampler = None

    if criterion is None:
        weight = bincount[0] / bincount[1] if positive_class_loss_weight is None else positive_class_loss_weight
        # weight = bincount[0] / bincount[1]
        criterion = torch.nn.BCEWithLogitsLoss(weight=weight) # Increase positive class weight
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=(training_sampler is None), sampler=training_sampler)
    val_loader = DataLoader(validation_data, batch_size=batch_size, sampler=val_sampler)

    all_training_predictions_logits = []
    all_val_predictions_logits = []
    training_y_batches = []
    val_y_batches = []

    for epoch, ((training_X_batch, training_y_batch), (val_X_batch, val_y_batch)) in enumerate(zip(train_loader, val_loader)):
        training_y_batches.append(training_y_batch)
        val_y_batches.append(val_y_batch)
        
        training_X_batch.to(device)
        val_X_batch.to(device)
        training_y_batch.to(device)
        val_y_batch.to(device)

        optimizer.zero_grad()
        training_predictions = model(training_X_batch, embedding_features_columns)
        all_training_predictions_logits.extend(training_predictions.detach())
        loss = criterion(training_predictions, training_y_batch)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            validation_predictions = model(val_X_batch, embedding_features_columns)
            all_val_predictions_logits.extend(validation_predictions.detach())
            validation_loss = criterion(validation_predictions, val_y_batch)

        training_losses.append(loss.detach())
        validation_losses.append(validation_loss.detach())


        epoch_accuracy = (logits_to_prediction(training_predictions) == training_y_batch).float().mean()
        epoch_val_accuracy = (logits_to_prediction(validation_predictions) == val_y_batch).float().mean()

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

    all_training_predictions = logits_to_prediction(torch.Tensor(all_training_predictions_logits))
    all_val_predictions = logits_to_prediction(torch.Tensor(all_val_predictions_logits))
    train_confusion_matrix = metrics.confusion_matrix(np.concatenate(training_y_batches), all_training_predictions)
    val_confusion_matrix = metrics.confusion_matrix(np.concatenate(val_y_batches), all_val_predictions)
    
    # Visualize sampling distribution
    positive_targets_per_batch_training = [sum(batch)[0] for batch in training_y_batches]
    negative_targets_per_batch_training = [len(batch)-positives for batch, positives in zip(training_y_batches, positive_targets_per_batch_training)]
    positive_targets_per_batch_val = [sum(batch)[0] for batch in val_y_batches]
    negative_targets_per_batch_val = [len(batch)-positives for batch, positives in zip(val_y_batches, positive_targets_per_batch_val)]

    bar_width = 0.35
    batches = np.array(range(1, len(training_y_batches)+1))
    plt.bar(batches, negative_targets_per_batch_training, bar_width, label="0")
    plt.bar(batches+bar_width, positive_targets_per_batch_training, bar_width, label="1")
    # plt.xticks(batches, batches + 1)
    plt.title("Sampling Proportions in Training Data")
    plt.legend()
    plt.show()
    plt.clf()

    bar_width = 0.35
    batches = np.array(range(1, len(val_y_batches)+1))
    plt.bar(batches, negative_targets_per_batch_val, bar_width, label="0")
    plt.bar(batches+bar_width, positive_targets_per_batch_val, bar_width, label="1")
    # plt.xticks(batches, batches + 1)
    plt.title("Sampling Proportions in Validation Data")
    plt.legend()
    plt.show()
    plt.clf()

    return optimal_model_dict, training_losses, validation_losses, training_accuracies, val_accuracies, train_confusion_matrix, val_confusion_matrix
    

        