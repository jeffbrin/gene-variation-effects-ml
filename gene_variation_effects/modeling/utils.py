import torch

def logits_to_prediction(logits: torch.Tensor, positive_threshold: float) -> torch.Tensor:
        return (torch.sigmoid(logits) >= positive_threshold).type(torch.int)