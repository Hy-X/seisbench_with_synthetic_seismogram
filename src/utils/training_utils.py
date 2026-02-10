"""
Utility functions for training and evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in seismic phase detection.
    
    Args:
        alpha: Weighting factor for each class
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, 
                 reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (batch_size, n_classes, n_timesteps)
            targets: Ground truth of shape (batch_size, n_classes, n_timesteps)
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets.argmax(dim=1), 
                                             reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets.argmax(dim=1).view(-1))
            focal_loss = at.view(targets.shape[0], -1) * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions of shape (batch_size, n_classes, n_timesteps)
            targets: Ground truth of shape (batch_size, n_classes, n_timesteps)
        """
        inputs = torch.softmax(inputs, dim=1)
        
        intersection = (inputs * targets).sum(dim=(0, 2))
        union = inputs.sum(dim=(0, 2)) + targets.sum(dim=(0, 2))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for seismic phase detection.
    Combines Cross-Entropy and Dice Loss.
    
    Args:
        ce_weight: Weight for cross-entropy loss
        dice_weight: Weight for dice loss
    """
    
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(inputs, targets.argmax(dim=1))
        dice = self.dice_loss(inputs, targets)
        return self.ce_weight * ce + self.dice_weight * dice


def calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor, 
                     num_classes: int = 3) -> Dict[str, float]:
    """
    Calculate evaluation metrics for phase detection.
    
    Args:
        predictions: Model predictions of shape (batch_size, n_classes, n_timesteps)
        targets: Ground truth of shape (batch_size, n_classes, n_timesteps)
        num_classes: Number of classes
        
    Returns:
        Dictionary containing precision, recall, and F1 score for each class
    """
    pred_labels = torch.argmax(predictions, dim=1).cpu().numpy().flatten()
    true_labels = torch.argmax(targets, dim=1).cpu().numpy().flatten()
    
    metrics = {}
    
    for i in range(num_classes):
        tp = np.sum((pred_labels == i) & (true_labels == i))
        fp = np.sum((pred_labels == i) & (true_labels != i))
        fn = np.sum((pred_labels != i) & (true_labels == i))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'class_{i}_precision'] = precision
        metrics[f'class_{i}_recall'] = recall
        metrics[f'class_{i}_f1'] = f1
    
    # Overall accuracy
    metrics['accuracy'] = np.mean(pred_labels == true_labels)
    
    return metrics


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
