"""
Training script for U-Net seismic phase picker.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json

from unet_model import create_unet
from label_generator import GaussianLabelGenerator


class SeismicDataset(Dataset):
    """PyTorch Dataset for seismic data."""
    
    def __init__(self, data_list, label_list):
        """
        Initialize dataset.
        
        Args:
            data_list: List of waveform arrays (n_channels, n_samples)
            label_list: List of label arrays (n_samples,)
        """
        self.data = data_list
        self.labels = label_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.FloatTensor(self.labels[idx]).unsqueeze(0)  # Add channel dimension
        return data, label


def train_model(model, train_loader, val_loader, 
                epochs=50, learning_rate=0.001,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                save_dir='models'):
    """
    Train the U-Net model.
    
    Args:
        model: U-Net model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save model checkpoints
        
    Returns:
        history: Dictionary with training history
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}")
        print(f"  LR:         {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_dir / 'best_model.pth')
            print(f"  Saved best model (val_loss: {avg_val_loss:.6f})")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
        
        print()
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, save_dir / 'final_model.pth')
    
    # Save training history
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def load_model(model, checkpoint_path, device='cpu'):
    """
    Load model from checkpoint.
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        epoch: Training epoch of checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    
    print(f"Loaded model from epoch {epoch}")
    if 'val_loss' in checkpoint:
        print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    return model, epoch


if __name__ == "__main__":
    # Example training workflow
    print("Training script example")
    print("="*70)
    
    # This is a placeholder - you would load your actual data here
    print("\nNote: This is a template. Replace with your actual data loading logic.")
    print("\nExample usage:")
    print("  1. Load your miniSEED files and create labels")
    print("  2. Split into train/validation sets")
    print("  3. Create DataLoaders")
    print("  4. Train the model using train_model()")
    print("  5. Load the best model using load_model()")
