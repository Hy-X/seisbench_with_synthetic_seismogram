"""
Inference script for seismic phase detection and catalog curation.
"""

import torch
import numpy as np
import argparse
import os
from typing import List, Dict, Tuple

from src.models import SeismicUNet
from src.data import SeismicDataset, get_dataloader, normalize_waveform
from src.utils import load_checkpoint


class PhasePicker:
    """
    Phase picker for seismic catalog curation.
    
    This class uses the trained U-Net model to detect P-wave and S-wave
    arrivals in seismic waveforms and outputs structured catalog information.
    """
    
    def __init__(self, model_path: str, n_channels: int = 3, n_classes: int = 3, 
                 device: str = 'cpu'):
        """
        Initialize the phase picker.
        
        Args:
            model_path: Path to trained model checkpoint
            n_channels: Number of input channels
            n_classes: Number of output classes
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = torch.device(device)
        self.n_classes = n_classes
        
        # Load model
        self.model = SeismicUNet(n_channels=n_channels, n_classes=n_classes)
        load_checkpoint(model_path, self.model)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def predict(self, waveform: np.ndarray) -> np.ndarray:
        """
        Predict phase labels for a waveform.
        
        Args:
            waveform: Input waveform of shape (n_channels, n_timesteps)
            
        Returns:
            Predicted labels of shape (n_classes, n_timesteps)
        """
        # Normalize waveform
        waveform = normalize_waveform(waveform, method='std')
        
        # Convert to tensor
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(waveform_tensor)
            probabilities = torch.softmax(output, dim=1)
        
        return probabilities.squeeze(0).cpu().numpy()
    
    def predict_batch(self, waveforms: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Predict phase labels for a batch of waveforms.
        
        Args:
            waveforms: Input waveforms of shape (n_samples, n_channels, n_timesteps)
            batch_size: Batch size for inference
            
        Returns:
            Predicted labels of shape (n_samples, n_classes, n_timesteps)
        """
        dataset = SeismicDataset(waveforms)
        dataloader = get_dataloader(dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=0)
        
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                output = self.model(batch)
                probabilities = torch.softmax(output, dim=1)
                predictions.append(probabilities.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def extract_phase_arrivals(self, probabilities: np.ndarray, 
                              threshold: float = 0.5,
                              sampling_rate: float = 100.0) -> Dict[str, List[float]]:
        """
        Extract phase arrival times from prediction probabilities.
        
        Args:
            probabilities: Prediction probabilities of shape (n_classes, n_timesteps)
            threshold: Threshold for phase detection
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Dictionary containing P-wave and S-wave arrival times
        """
        arrivals = {'P': [], 'S': []}
        
        # P-wave detection (class 1)
        p_probs = probabilities[1, :]
        p_peaks = self._find_peaks(p_probs, threshold)
        arrivals['P'] = [peak / sampling_rate for peak in p_peaks]
        
        # S-wave detection (class 2)
        s_probs = probabilities[2, :]
        s_peaks = self._find_peaks(s_probs, threshold)
        arrivals['S'] = [peak / sampling_rate for peak in s_peaks]
        
        return arrivals
    
    @staticmethod
    def _find_peaks(signal: np.ndarray, threshold: float, 
                   min_distance: int = 100) -> List[int]:
        """
        Find peaks in a signal.
        
        Args:
            signal: 1D signal array
            threshold: Minimum threshold for peak detection
            min_distance: Minimum distance between peaks
            
        Returns:
            List of peak indices
        """
        peaks = []
        i = 0
        while i < len(signal):
            if signal[i] > threshold:
                # Find local maximum in window
                window_end = min(i + min_distance, len(signal))
                peak_idx = i + np.argmax(signal[i:window_end])
                peaks.append(peak_idx)
                i = peak_idx + min_distance
            else:
                i += 1
        return peaks
    
    def create_catalog_entry(self, waveform: np.ndarray, 
                           station_name: str,
                           start_time: str,
                           threshold: float = 0.5,
                           sampling_rate: float = 100.0) -> Dict:
        """
        Create a catalog entry for a waveform.
        
        Args:
            waveform: Input waveform
            station_name: Station identifier
            start_time: Start time of the waveform (ISO format)
            threshold: Detection threshold
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Catalog entry dictionary
        """
        probabilities = self.predict(waveform)
        arrivals = self.extract_phase_arrivals(probabilities, threshold, sampling_rate)
        
        entry = {
            'station': station_name,
            'start_time': start_time,
            'sampling_rate': sampling_rate,
            'p_arrivals': arrivals['P'],
            's_arrivals': arrivals['S'],
            'has_p': len(arrivals['P']) > 0,
            'has_s': len(arrivals['S']) > 0
        }
        
        return entry


def main():
    parser = argparse.ArgumentParser(description='Run inference for seismic phase detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input data (numpy array)')
    parser.add_argument('--output', type=str, default='outputs/predictions',
                       help='Output directory for predictions')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Detection threshold')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize phase picker
    picker = PhasePicker(args.model, device=args.device)
    
    # Load data
    print(f"Loading data from {args.input}")
    data = np.load(args.input)
    print(f"Data shape: {data.shape}")
    
    # Run inference
    print("Running inference...")
    predictions = picker.predict_batch(data, batch_size=args.batch_size)
    
    # Save predictions
    output_path = os.path.join(args.output, 'predictions.npy')
    np.save(output_path, predictions)
    print(f"Saved predictions to {output_path}")
    
    # Extract phase arrivals for first sample as example
    if len(predictions) > 0:
        arrivals = picker.extract_phase_arrivals(
            predictions[0], 
            threshold=args.threshold
        )
        print(f"\nExample arrivals for first sample:")
        print(f"P-wave arrivals: {arrivals['P']}")
        print(f"S-wave arrivals: {arrivals['S']}")


if __name__ == '__main__':
    main()
