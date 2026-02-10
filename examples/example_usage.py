"""
Example script demonstrating the usage of the U-Net for seismic phase detection.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import SeismicUNet
from src.data import create_synthetic_data, normalize_waveform


def visualize_prediction(waveform, prediction, save_path=None):
    """
    Visualize waveform and phase predictions.
    
    Args:
        waveform: Input waveform of shape (n_channels, n_timesteps)
        prediction: Prediction probabilities of shape (n_classes, n_timesteps)
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    
    # Plot waveform channels
    for i in range(min(3, waveform.shape[0])):
        axes[0].plot(waveform[i], label=f'Channel {i}', alpha=0.7)
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Input Waveform')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot prediction probabilities
    labels = ['Noise', 'P-wave', 'S-wave']
    for i in range(prediction.shape[0]):
        axes[i+1].plot(prediction[i], label=labels[i])
        axes[i+1].set_ylabel('Probability')
        axes[i+1].set_title(f'{labels[i]} Probability')
        axes[i+1].legend()
        axes[i+1].grid(True)
        axes[i+1].set_ylim([0, 1])
    
    axes[-1].set_xlabel('Time Steps')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def example_training():
    """Example of training the U-Net model."""
    print("=" * 50)
    print("Example 1: Training U-Net")
    print("=" * 50)
    
    # Create synthetic data
    print("\n1. Creating synthetic data...")
    data, labels = create_synthetic_data(n_samples=100, n_channels=3, 
                                         n_timesteps=3000, n_classes=3)
    print(f"   Data shape: {data.shape}")
    print(f"   Labels shape: {labels.shape}")
    
    # Initialize model
    print("\n2. Initializing model...")
    model = SeismicUNet(n_channels=3, n_classes=3, dropout=0.1)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare data
    print("\n3. Preparing data batch...")
    batch_data = torch.from_numpy(data[:8]).float()
    batch_labels = torch.from_numpy(labels[:8]).float()
    print(f"   Batch data shape: {batch_data.shape}")
    
    # Forward pass
    print("\n4. Running forward pass...")
    model.eval()
    with torch.no_grad():
        predictions = model(batch_data)
        probabilities = torch.softmax(predictions, dim=1)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Probabilities shape: {probabilities.shape}")
    
    print("\n✓ Training example completed successfully!")


def example_inference():
    """Example of using the trained model for inference."""
    print("\n" + "=" * 50)
    print("Example 2: Inference with U-Net")
    print("=" * 50)
    
    # Create model
    print("\n1. Initializing model...")
    model = SeismicUNet(n_channels=3, n_classes=3)
    model.eval()
    
    # Create a synthetic waveform
    print("\n2. Creating synthetic waveform...")
    data, _ = create_synthetic_data(n_samples=1, n_channels=3, 
                                    n_timesteps=3000, n_classes=3)
    waveform = data[0]
    print(f"   Waveform shape: {waveform.shape}")
    
    # Normalize
    waveform_norm = normalize_waveform(waveform, method='std')
    
    # Run inference
    print("\n3. Running inference...")
    waveform_tensor = torch.from_numpy(waveform_norm).float().unsqueeze(0)
    with torch.no_grad():
        output = model(waveform_tensor)
        probabilities = torch.softmax(output, dim=1)
    
    predictions = probabilities.squeeze(0).numpy()
    print(f"   Predictions shape: {predictions.shape}")
    
    # Find phase arrivals
    print("\n4. Detecting phase arrivals...")
    p_arrivals = np.where(predictions[1] > 0.5)[0]
    s_arrivals = np.where(predictions[2] > 0.5)[0]
    
    if len(p_arrivals) > 0:
        print(f"   P-wave detected at time steps: {p_arrivals[:5]}...")
    else:
        print("   No P-wave detected")
    
    if len(s_arrivals) > 0:
        print(f"   S-wave detected at time steps: {s_arrivals[:5]}...")
    else:
        print("   No S-wave detected")
    
    print("\n✓ Inference example completed successfully!")
    
    return waveform, predictions


def example_catalog_curation():
    """Example of catalog curation workflow."""
    print("\n" + "=" * 50)
    print("Example 3: Catalog Curation")
    print("=" * 50)
    
    # Create synthetic catalog data
    print("\n1. Creating synthetic catalog data...")
    n_events = 10
    data, _ = create_synthetic_data(n_samples=n_events, n_channels=3,
                                    n_timesteps=3000, n_classes=3)
    
    # Initialize model
    print("\n2. Initializing model for catalog curation...")
    model = SeismicUNet(n_channels=3, n_classes=3)
    model.eval()
    
    # Process each event
    print(f"\n3. Processing {n_events} events...")
    catalog = []
    
    for i in range(n_events):
        waveform = normalize_waveform(data[i], method='std')
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        
        with torch.no_grad():
            output = model(waveform_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        predictions = probabilities.squeeze(0).numpy()
        
        # Extract phase information
        p_detected = np.any(predictions[1] > 0.5)
        s_detected = np.any(predictions[2] > 0.5)
        
        catalog_entry = {
            'event_id': i,
            'has_p_wave': p_detected,
            'has_s_wave': s_detected,
            'max_p_probability': float(np.max(predictions[1])),
            'max_s_probability': float(np.max(predictions[2]))
        }
        catalog.append(catalog_entry)
    
    # Print catalog summary
    print("\n4. Catalog Summary:")
    print(f"   Total events: {len(catalog)}")
    print(f"   Events with P-wave: {sum(e['has_p_wave'] for e in catalog)}")
    print(f"   Events with S-wave: {sum(e['has_s_wave'] for e in catalog)}")
    print(f"   Events with both: {sum(e['has_p_wave'] and e['has_s_wave'] for e in catalog)}")
    
    # Show first few entries
    print("\n   First 3 catalog entries:")
    for entry in catalog[:3]:
        print(f"   - Event {entry['event_id']}: "
              f"P={entry['has_p_wave']}, S={entry['has_s_wave']}, "
              f"P_prob={entry['max_p_probability']:.3f}, "
              f"S_prob={entry['max_s_probability']:.3f}")
    
    print("\n✓ Catalog curation example completed successfully!")


def main():
    """Run all examples."""
    print("\n" + "=" * 50)
    print("U-Net Seismic Phase Detection - Examples")
    print("=" * 50)
    
    # Example 1: Training
    example_training()
    
    # Example 2: Inference
    waveform, predictions = example_inference()
    
    # Example 3: Catalog curation
    example_catalog_curation()
    
    # Visualize one example
    print("\n" + "=" * 50)
    print("Generating visualization...")
    print("=" * 50)
    visualize_prediction(waveform, predictions, save_path='outputs/example_prediction.png')
    
    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)


if __name__ == '__main__':
    import os
    os.makedirs('outputs', exist_ok=True)
    main()
