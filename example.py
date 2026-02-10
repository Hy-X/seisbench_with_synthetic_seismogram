"""
Example script demonstrating the complete workflow:
1. Read miniSEED files
2. Load/create U-Net model
3. Generate Gaussian labels
4. Make predictions
5. Evaluate with 1-second tolerance
6. Visualize results and statistics
"""

import numpy as np
import torch
import os
from pathlib import Path

from data_loader import MiniSEEDLoader
from unet_model import create_unet
from label_generator import GaussianLabelGenerator
from evaluation import evaluate_predictions, extract_picks_from_prediction
from visualization import (plot_waveform_with_predictions, 
                          plot_multiple_examples, 
                          plot_statistics,
                          print_statistics)


def create_synthetic_data(n_samples=2000, n_channels=3, n_picks=2):
    """
    Create synthetic 3-channel seismic data for demonstration.
    
    Args:
        n_samples: Number of samples (default: 2000 for 20s at 100 Hz)
        n_channels: Number of channels (default: 3)
        n_picks: Number of phase arrivals
        
    Returns:
        data: Synthetic waveform data
        pick_samples: List of pick locations
    """
    # Generate random picks
    pick_samples = np.random.randint(n_samples // 4, 3 * n_samples // 4, size=n_picks)
    pick_samples = sorted(pick_samples)
    
    # Create synthetic data
    data = np.random.randn(n_channels, n_samples) * 0.1
    
    # Add signals at pick locations
    for pick in pick_samples:
        # Add arrival signal
        arrival_length = 200
        start = max(0, pick - 20)
        end = min(n_samples, pick + arrival_length)
        
        # Create synthetic arrival with exponential decay
        signal_length = end - start
        t = np.arange(signal_length)
        envelope = np.exp(-t / 50.0)
        
        for ch in range(n_channels):
            # Add oscillating signal with decay
            freq = 5 + ch * 2  # Different frequency per channel
            signal = envelope * np.sin(2 * np.pi * freq * t / 100.0)
            data[ch, start:end] += signal * (1.0 + ch * 0.2)
    
    return data, pick_samples


def main():
    """Main demonstration function."""
    
    print("\n" + "="*70)
    print("SEISMIC PHASE PICKING WITH U-NET")
    print("="*70 + "\n")
    
    # Configuration
    sample_rate = 100.0  # Hz
    window_length = 20.0  # seconds
    n_samples = int(sample_rate * window_length)
    sigma = 50  # samples for Gaussian label
    tolerance_samples = 100  # 1 second tolerance at 100 Hz
    
    print("Configuration:")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Window Length: {window_length} s")
    print(f"  Total Samples: {n_samples}")
    print(f"  Label Sigma: {sigma} samples")
    print(f"  Tolerance: {tolerance_samples} samples ({tolerance_samples/sample_rate:.1f}s)")
    print()
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}/")
    print()
    
    # Step 1: Create or load data
    print("Step 1: Creating synthetic data...")
    print("-" * 70)
    
    n_examples = 5
    data_list = []
    label_list = []
    pick_list = []
    
    for i in range(n_examples):
        # Generate synthetic data
        data, picks = create_synthetic_data(n_samples=n_samples, n_picks=2)
        data_list.append(data)
        pick_list.append(picks)
        
        # Generate Gaussian labels
        label_gen = GaussianLabelGenerator(sigma=sigma, sample_rate=sample_rate)
        pick_times = picks / sample_rate
        label = label_gen(n_samples, pick_times)
        label_list.append(label)
        
        print(f"  Example {i+1}: {len(picks)} picks at samples {picks}")
    
    print(f"\nGenerated {n_examples} synthetic examples")
    print()
    
    # Step 2: Create U-Net model
    print("Step 2: Creating U-Net model...")
    print("-" * 70)
    
    model = create_unet(variant='standard', in_channels=3, out_channels=1)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model variant: standard")
    print(f"  Input channels: 3")
    print(f"  Output channels: 1")
    print(f"  Total parameters: {n_params:,}")
    print()
    
    # Step 3: Make predictions (with untrained model for demonstration)
    print("Step 3: Making predictions...")
    print("-" * 70)
    print("  Note: Using untrained model for demonstration")
    print("  In practice, you would train the model first")
    print()
    
    model.eval()
    prediction_list = []
    
    with torch.no_grad():
        for i, data in enumerate(data_list):
            # Convert to tensor
            data_tensor = torch.FloatTensor(data).unsqueeze(0)  # Add batch dimension
            
            # Predict
            pred = model(data_tensor)
            pred = pred.squeeze().cpu().numpy()
            prediction_list.append(pred)
            
            # Extract picks
            pred_picks = extract_picks_from_prediction(pred, threshold=0.5)
            true_picks = pick_list[i]
            
            print(f"  Example {i+1}:")
            print(f"    True picks: {true_picks}")
            print(f"    Predicted picks: {pred_picks}")
    
    print()
    
    # Step 4: Evaluate predictions
    print("Step 4: Evaluating predictions...")
    print("-" * 70)
    
    # Stack for batch evaluation
    predictions_batch = np.stack(prediction_list, axis=0)
    labels_batch = np.stack(label_list, axis=0)
    
    metrics = evaluate_predictions(
        predictions_batch,
        labels_batch,
        threshold=0.5,
        tolerance_samples=tolerance_samples,
        sample_rate=sample_rate
    )
    
    print_statistics(metrics)
    
    # Step 5: Visualize results
    print("Step 5: Visualizing results...")
    print("-" * 70)
    
    # Plot individual examples
    n_plot = min(3, n_examples)
    for i in range(n_plot):
        save_path = output_dir / f"example_{i+1}.png"
        print(f"  Plotting example {i+1}...")
        
        plot_waveform_with_predictions(
            data_list[i],
            label_list[i],
            prediction_list[i],
            channel_names=['Z', 'N', 'E'],
            sample_rate=sample_rate,
            threshold=0.5,
            save_path=str(save_path)
        )
    
    # Plot statistics
    stats_path = output_dir / "statistics.png"
    print(f"  Plotting statistics...")
    plot_statistics(metrics, save_path=str(stats_path))
    
    print()
    print("="*70)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"\nResults saved in: {output_dir}/")
    print(f"  - {n_plot} example plots")
    print(f"  - 1 statistics plot")
    print()


if __name__ == "__main__":
    main()
