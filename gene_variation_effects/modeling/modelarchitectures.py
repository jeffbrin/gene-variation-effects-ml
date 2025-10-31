import torch 
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, embedding_dimension_mapping : np.ndarray, hidden_sizes : list[int], input_size : int) -> None:
        """
        Initalizes neural network and embedding layers for high cardinality categorical variables
        
        Args:
            embedding_dimension_mapping (np.ndarray): 2d array of shape (FEATURES, 2) where we have features that require embedding as rows, number
            of unique categories as index 0 of columns and dimension of output embedding as index 1. Order of features as rows is:
            'GeneSymbol', 'Cytogenetic', 'ReferenceAlleleVCF', and 'AlternateAlleleVCF'
            hidden_size (list[int]) : Length of the list indicates the number of layers and int values says number of neurons excluding input and output layers
            input_size (int) : size of input layer after embedding of high cardinality categories #TODO probably a better way to do this instead of having it as input. Maybe calculate the input dim in the init
        """        
        super().__init__()
        self.embedding_networks = []
        EMB_INPUT_DIMENSION_INDEX = 0
        EMB_OUTPUT_DIMENSION_INDEX = 1
        for feature in embedding_dimension_mapping:
            self.embedding_networks.append(nn.Embedding(num_embeddings = feature[EMB_INPUT_DIMENSION_INDEX], 
                                                        embedding_dim = feature[EMB_OUTPUT_DIMENSION_INDEX]))
        
        # Remove a column for each category and add the embedding output size
        input_size = input_size + sum(x[EMB_OUTPUT_DIMENSION_INDEX] for x in embedding_dimension_mapping) - embedding_dimension_mapping.shape[0]
        self.layers = [nn.Linear(input_size, hidden_sizes[0])]
        for layer in range(len(hidden_sizes)):
            # For before output layer, we have n in neurons and 1 out neuron as we're doing binary classification
            if layer == len(hidden_sizes) - 1:
                self.layers.append(nn.Linear(hidden_sizes[layer], 1))
            else:
                self.layers.append(nn.Linear(hidden_sizes[layer], hidden_sizes[layer + 1]))
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        pass
