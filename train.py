"""
Training script for U-Net seismic phase detection model.
"""

import torch
import torch.optim as optim
from torch.utils.data import random_split
import argparse
import yaml
import os
from tqdm import tqdm

from src.models import SeismicUNet
from src.data import create_synthetic_data, SeismicDataset, get_dataloader
from src.utils import CombinedLoss, calculate_metrics, save_checkpoint


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(dataloader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            metrics = calculate_metrics(output, target)
            all_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
    
    return total_loss / len(dataloader), avg_metrics


def train(config):
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create synthetic data (replace with real data loading)
    print("Creating synthetic data...")
    data, labels = create_synthetic_data(
        n_samples=config['n_samples'],
        n_channels=config['n_channels'],
        n_timesteps=config['n_timesteps'],
        n_classes=config['n_classes']
    )
    
    # Create dataset
    dataset = SeismicDataset(data, labels)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config['num_workers']
    )
    val_loader = get_dataloader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=config['num_workers']
    )
    
    # Initialize model
    model = SeismicUNet(
        n_channels=config['n_channels'],
        n_classes=config['n_classes'],
        dropout=config['dropout']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(
        ce_weight=config['ce_weight'],
        dice_weight=config['dice_weight']
    )
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config['output_dir'], 'best_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['save_every'] == 0:
            checkpoint_path = os.path.join(config['output_dir'], f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for seismic phase detection')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'n_samples': 1000,
            'n_channels': 3,
            'n_timesteps': 3000,
            'n_classes': 3,
            'batch_size': 16,
            'num_workers': 4,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'ce_weight': 0.5,
            'dice_weight': 0.5,
            'num_epochs': 50,
            'save_every': 10,
            'output_dir': 'outputs/models'
        }
        print(f"Config file not found at {args.config}, using default configuration")
    
    train(config)


if __name__ == '__main__':
    main()
