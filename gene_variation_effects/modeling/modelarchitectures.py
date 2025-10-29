import torch 
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, size_dim_emb : np.ndarray, hidden_sizes : list[int], input_size : int) -> None:
        """
        Initalizes neural network and embedding layers for high cardinality categorical variables
        
        Args:
            size_dim_emb (np.ndarray): 2d array of shape (4, 2) where we have features that require embedding as rows, number
            of unique categories as index 0 of columns and dimension of output embedding as index 1. Order of features as rows is:
            'GeneSymbol', 'Cytogenetic', 'ReferenceAlleleVCF', and 'AlternateAlleleVCF'
            hidden_size (list[int]) : Length of the list indicates the number of layers and int values says number of neurons excluding input and output layers
            input_size (int) : size of input layer after embedding of high cardinality categories #TODO probably a better way to do this instead of having it as input. Maybe calculate the input dim in the init
        """        
        super().__init__()
        self.embedding_networks = []
        for feature in range(4):
            self.embedding_networks.append(nn.Embedding(num_embeddings = size_dim_emb[feature, 0], embedding_dim = size_dim_emb[feature, 1]))
        
        self.layers = [nn.Linear(input_size, hidden_sizes[0])]
        for layer in range(len(hidden_sizes)):
            # For before output layer, we have n in neurons and 1 out neuron as we're doing binary classification
            if layer == len(hidden_sizes - 1):
                self.layers.append(nn.Linear(hidden_sizes[layer], 1))
            else:
                self.layers.append(nn.Linear(hidden_sizes[layer], hidden_sizes[layer + 1]))
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        pass
