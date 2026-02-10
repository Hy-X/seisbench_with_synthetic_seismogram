"""
Utility functions for batch processing miniSEED files.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json
import torch

from data_loader import MiniSEEDLoader
from label_generator import GaussianLabelGenerator
from evaluation import evaluate_predictions, extract_picks_from_prediction
from visualization import plot_waveform_with_predictions


def process_mseed_directory(mseed_dir: str,
                            catalog_file: Optional[str] = None,
                            output_dir: str = 'processed',
                            sample_rate: float = 100.0,
                            window_length: float = 20.0,
                            sigma: float = 50.0) -> Tuple[List, List, List]:
    """
    Process all miniSEED files in a directory.
    
    Args:
        mseed_dir: Directory containing miniSEED files
        catalog_file: Optional JSON file with pick catalog
        output_dir: Directory to save processed data
        sample_rate: Target sampling rate
        window_length: Window length in seconds
        sigma: Gaussian label sigma
        
    Returns:
        data_list: List of waveform arrays
        label_list: List of label arrays (if catalog provided)
        file_list: List of processed filenames
    """
    mseed_dir = Path(mseed_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load catalog if provided
    catalog = {}
    if catalog_file:
        with open(catalog_file, 'r') as f:
            catalog = json.load(f)
    
    # Initialize loader and label generator
    loader = MiniSEEDLoader(target_sample_rate=sample_rate, window_length=window_length)
    label_gen = GaussianLabelGenerator(sigma=sigma, sample_rate=sample_rate)
    
    data_list = []
    label_list = []
    file_list = []
    
    # Find all miniSEED files
    mseed_files = sorted(mseed_dir.glob('*.mseed')) + sorted(mseed_dir.glob('*.MSEED'))
    
    print(f"Found {len(mseed_files)} miniSEED files")
    
    for mseed_file in mseed_files:
        try:
            # Read data
            data, channel_names = loader.read_mseed(str(mseed_file), normalize=True)
            data_list.append(data)
            file_list.append(mseed_file.name)
            
            # Generate label if catalog available
            if catalog and mseed_file.name in catalog:
                picks = catalog[mseed_file.name]
                label = label_gen(loader.n_samples, picks)
                label_list.append(label)
            else:
                # Create empty label
                label_list.append(np.zeros(loader.n_samples, dtype=np.float32))
            
            print(f"  Processed: {mseed_file.name}")
            
        except Exception as e:
            print(f"  Error processing {mseed_file.name}: {str(e)}")
            continue
    
    return data_list, label_list, file_list


def batch_predict(model, data_list: List[np.ndarray],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 batch_size: int = 32) -> List[np.ndarray]:
    """
    Make predictions on a batch of waveforms.
    
    Args:
        model: Trained U-Net model
        data_list: List of waveform arrays
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        predictions: List of prediction arrays
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i+batch_size]
            batch_tensor = torch.FloatTensor(np.stack(batch_data, axis=0)).to(device)
            
            batch_pred = model(batch_tensor)
            batch_pred = batch_pred.cpu().numpy().squeeze()
            
            if batch_pred.ndim == 1:
                predictions.append(batch_pred)
            else:
                predictions.extend(list(batch_pred))
    
    return predictions


def save_picks_to_catalog(predictions: List[np.ndarray],
                         file_list: List[str],
                         output_file: str,
                         threshold: float = 0.5,
                         sample_rate: float = 100.0):
    """
    Save extracted picks to a JSON catalog file.
    
    Args:
        predictions: List of prediction arrays
        file_list: List of filenames
        output_file: Output JSON file path
        threshold: Detection threshold
        sample_rate: Sampling rate
    """
    catalog = {}
    
    for pred, filename in zip(predictions, file_list):
        pick_samples = extract_picks_from_prediction(pred, threshold=threshold)
        pick_times = (pick_samples / sample_rate).tolist()
        catalog[filename] = pick_times
    
    with open(output_file, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"Saved picks for {len(catalog)} files to {output_file}")


def batch_visualize(data_list: List[np.ndarray],
                   label_list: List[np.ndarray],
                   prediction_list: List[np.ndarray],
                   file_list: List[str],
                   output_dir: str = 'visualizations',
                   n_examples: Optional[int] = None,
                   sample_rate: float = 100.0):
    """
    Create visualizations for multiple examples.
    
    Args:
        data_list: List of waveform arrays
        label_list: List of label arrays
        prediction_list: List of prediction arrays
        file_list: List of filenames
        output_dir: Output directory for plots
        n_examples: Number of examples to plot (None = all)
        sample_rate: Sampling rate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    n_examples = n_examples or len(data_list)
    n_examples = min(n_examples, len(data_list))
    
    for i in range(n_examples):
        output_file = output_dir / f"{Path(file_list[i]).stem}_plot.png"
        
        plot_waveform_with_predictions(
            data_list[i],
            label_list[i],
            prediction_list[i],
            channel_names=['Z', 'N', 'E'],
            sample_rate=sample_rate,
            save_path=str(output_file)
        )
        
        print(f"  Saved plot: {output_file.name}")


def create_catalog_template(mseed_dir: str, output_file: str = 'catalog_template.json'):
    """
    Create a template catalog file for manual pick annotation.
    
    Args:
        mseed_dir: Directory containing miniSEED files
        output_file: Output template file
    """
    mseed_dir = Path(mseed_dir)
    mseed_files = sorted(mseed_dir.glob('*.mseed')) + sorted(mseed_dir.glob('*.MSEED'))
    
    catalog = {}
    for mseed_file in mseed_files:
        catalog[mseed_file.name] = []  # Empty list for manual picks
    
    with open(output_file, 'w') as f:
        json.dump(catalog, f, indent=2)
    
    print(f"Created template catalog with {len(catalog)} files: {output_file}")
    print("Edit this file to add pick times in seconds.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch process miniSEED files')
    parser.add_argument('mseed_dir', help='Directory containing miniSEED files')
    parser.add_argument('--catalog', help='JSON file with pick catalog')
    parser.add_argument('--output', default='processed', help='Output directory')
    parser.add_argument('--create-template', action='store_true',
                       help='Create a catalog template file')
    
    args = parser.parse_args()
    
    if args.create_template:
        create_catalog_template(args.mseed_dir)
    else:
        data_list, label_list, file_list = process_mseed_directory(
            args.mseed_dir,
            catalog_file=args.catalog,
            output_dir=args.output
        )
        print(f"\nProcessed {len(data_list)} files successfully")
