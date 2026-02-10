"""
Plot seismograms from SeisBench-formatted dataset.

This script loads waveform data from a SeisBench-compatible dataset and
provides various visualization options including single trace plots with
phase arrivals and multi-trace comparison plots.

Usage:
    python plot_seismograms.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
from pathlib import Path
from typing import Optional, List, Tuple, Union
import warnings


# Configuration constants
DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLE_RATE = 100.0  # Hz
FIGURE_DPI = 100
DEFAULT_COMPONENT_ORDER = "ZNE"


def load_dataset(
    data_path: Union[str, Path],
    component_order: Optional[str] = None,
    sampling_rate: Optional[float] = None,
    cache: Optional[str] = None
) -> sbd.WaveformDataset:
    """
    Load a SeisBench WaveformDataset from disk.
    
    Args:
        data_path: Path to directory containing metadata.csv and waveforms.hdf5
        component_order: Order of seismic components (e.g., "ZNE", "ENZ").
                        If None, dataset will use default/existing order
        sampling_rate: Target sampling rate in Hz. If None, uses original rate
        cache: Caching strategy - "trace" or "full". None for no caching
        
    Returns:
        Loaded WaveformDataset instance
        
    Raises:
        FileNotFoundError: If dataset files are not found
        ValueError: If data_path is invalid
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")
    
    metadata_file = data_path / "metadata.csv"
    waveforms_file = data_path / "waveforms.hdf5"
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.csv not found in {data_path}")
    if not waveforms_file.exists():
        raise FileNotFoundError(f"waveforms.hdf5 not found in {data_path}")
    
    try:
        # Build kwargs conditionally
        kwargs = {
            'path': str(data_path),
            'sampling_rate': sampling_rate,
            'cache': cache
        }
        
        # Check if metadata has component_order column before setting it
        import pandas as pd
        metadata = pd.read_csv(metadata_file, nrows=1)
        
        if 'trace_component_order' in metadata.columns and component_order:
            kwargs['component_order'] = component_order
        elif component_order:
            warnings.warn(
                f"Component order '{component_order}' requested but metadata lacks "
                "'trace_component_order' column. Loading without component reordering."
            )
        
        dataset = sbd.WaveformDataset(**kwargs)
        
        print(f"✓ Successfully loaded dataset from {data_path}")
        print(f"  Total traces: {len(dataset)}")
        if component_order and 'component_order' in kwargs:
            print(f"  Component order: {component_order}")
        else:
            print(f"  Component order: Native (not reordered)")
        if sampling_rate:
            print(f"  Sampling rate: {sampling_rate} Hz")
        
        return dataset
        
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


def plot_single_seismogram(
    dataset: sbd.WaveformDataset,
    trace_idx: int,
    show_phase_arrivals: bool = True,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot a single three-component seismogram with optional phase arrivals.
    
    Creates a multi-panel plot showing Z, N, and E components with
    P and S wave arrival markers if available in metadata.
    
    Args:
        dataset: SeisBench WaveformDataset
        trace_idx: Index of trace to plot
        show_phase_arrivals: Whether to mark P and S wave arrivals
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save figure. If None, displays interactively
        
    Returns:
        Matplotlib figure object
        
    Raises:
        IndexError: If trace_idx is out of range
        ValueError: If waveform data is invalid
    """
    if trace_idx < 0 or trace_idx >= len(dataset):
        raise IndexError(
            f"Trace index {trace_idx} out of range [0, {len(dataset)-1}]"
        )
    
    # Get waveform data
    waveforms = dataset.get_waveforms(trace_idx)
    metadata = dataset.metadata.iloc[trace_idx]
    
    if waveforms.ndim != 2 or waveforms.shape[0] != 3:
        raise ValueError(
            f"Expected 3-component waveform, got shape {waveforms.shape}"
        )
    
    # Get sampling rate
    sample_rate = metadata.get('trace_sampling_rate_hz', SAMPLE_RATE)
    n_samples = waveforms.shape[1]
    time = np.arange(n_samples) / sample_rate
    
    # Extract phase arrivals
    p_sample = metadata.get('trace_p_arrival_sample', None)
    s_sample = metadata.get('trace_s_arrival_sample', None)
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        f"Seismogram: {metadata['trace_name']}\n"
        f"Station: {metadata['station_code']}.{metadata['network_code']} | "
        f"SNR: {metadata.get('snr_db', 'N/A'):.2f} dB",
        fontsize=14,
        fontweight='bold'
    )
    
    component_labels = ['Vertical (Z)', 'North (N)', 'East (E)']
    colors = ['black', 'darkblue', 'darkred']
    
    for i, (ax, label, color) in enumerate(zip(axes, component_labels, colors)):
        # Plot waveform
        ax.plot(time, waveforms[i], color=color, linewidth=0.5)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(time[0], time[-1])
        
        # Mark phase arrivals
        if show_phase_arrivals:
            if p_sample is not None and not np.isnan(p_sample):
                p_time = p_sample / sample_rate
                ax.axvline(
                    p_time, color='blue', linestyle='--', 
                    linewidth=2, alpha=0.7, label='P-wave'
                )
            
            if s_sample is not None and not np.isnan(s_sample):
                s_time = s_sample / sample_rate
                ax.axvline(
                    s_time, color='red', linestyle='--', 
                    linewidth=2, alpha=0.7, label='S-wave'
                )
            
            if i == 0:  # Add legend only to top panel
                ax.legend(loc='upper right', fontsize=10)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_multiple_seismograms(
    dataset: sbd.WaveformDataset,
    trace_indices: List[int],
    component: int = 0,
    normalize: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot multiple seismograms for comparison.
    
    Displays multiple traces on the same time axis, optionally normalized
    for amplitude comparison. Shows one component only.
    
    Args:
        dataset: SeisBench WaveformDataset
        trace_indices: List of trace indices to plot
        component: Which component to plot (0=Z, 1=N, 2=E)
        normalize: Whether to normalize each trace to unit amplitude
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save figure. If None, displays interactively
        
    Returns:
        Matplotlib figure object
        
    Raises:
        ValueError: If trace_indices is empty or component is invalid
    """
    if not trace_indices:
        raise ValueError("trace_indices cannot be empty")
    if component < 0 or component > 2:
        raise ValueError(f"component must be 0, 1, or 2, got {component}")
    
    n_traces = len(trace_indices)
    component_names = {0: 'Vertical (Z)', 1: 'North (N)', 2: 'East (E)'}
    
    fig, axes = plt.subplots(n_traces, 1, figsize=figsize, sharex=True)
    if n_traces == 1:
        axes = [axes]
    
    fig.suptitle(
        f"Multiple Seismogram Comparison - {component_names[component]} Component",
        fontsize=14,
        fontweight='bold'
    )
    
    for idx, (ax, trace_idx) in enumerate(zip(axes, trace_indices)):
        try:
            # Get waveform and metadata
            waveforms = dataset.get_waveforms(trace_idx)
            metadata = dataset.metadata.iloc[trace_idx]
            
            sample_rate = metadata.get('trace_sampling_rate_hz', SAMPLE_RATE)
            n_samples = waveforms.shape[1]
            time = np.arange(n_samples) / sample_rate
            
            # Extract component
            trace_data = waveforms[component]
            
            # Normalize if requested
            if normalize:
                max_amp = np.abs(trace_data).max()
                if max_amp > 0:
                    trace_data = trace_data / max_amp
            
            # Plot
            ax.plot(time, trace_data, 'k-', linewidth=0.5)
            ax.set_ylabel(
                f"{metadata['trace_name']}\n"
                f"SNR: {metadata.get('snr_db', 'N/A'):.1f} dB",
                fontsize=9
            )
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(time[0], time[-1])
            
            # Mark phase arrivals
            p_sample = metadata.get('trace_p_arrival_sample', None)
            s_sample = metadata.get('trace_s_arrival_sample', None)
            
            if p_sample is not None and not np.isnan(p_sample):
                p_time = p_sample / sample_rate
                ax.axvline(p_time, color='blue', linestyle='--', 
                          linewidth=1.5, alpha=0.6)
            
            if s_sample is not None and not np.isnan(s_sample):
                s_time = s_sample / sample_rate
                ax.axvline(s_time, color='red', linestyle='--', 
                          linewidth=1.5, alpha=0.6)
                
        except Exception as e:
            warnings.warn(f"Failed to plot trace {trace_idx}: {e}")
            ax.text(0.5, 0.5, f"Error loading trace {trace_idx}", 
                   ha='center', va='center', transform=ax.transAxes)
    
    axes[-1].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    
    # Add custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', linestyle='--', linewidth=2, label='P-wave'),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='S-wave')
    ]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def display_dataset_info(dataset: sbd.WaveformDataset) -> None:
    """
    Display summary information about the dataset.
    
    Args:
        dataset: SeisBench WaveformDataset
    """
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total traces: {len(dataset)}")
    print(f"Metadata columns: {list(dataset.metadata.columns)}")
    
    # Get statistics
    metadata = dataset.metadata
    
    if 'trace_sampling_rate_hz' in metadata.columns:
        sample_rates = metadata['trace_sampling_rate_hz'].unique()
        print(f"Sampling rate(s): {sample_rates} Hz")
    
    if 'trace_npts' in metadata.columns:
        n_samples = metadata['trace_npts'].unique()
        durations = n_samples / SAMPLE_RATE
        print(f"Trace duration(s): {durations} seconds")
    
    if 'snr_db' in metadata.columns:
        snr_stats = metadata['snr_db'].describe()
        print(f"\nSNR statistics:")
        print(f"  Mean: {snr_stats['mean']:.2f} dB")
        print(f"  Median: {snr_stats['50%']:.2f} dB")
        print(f"  Range: [{snr_stats['min']:.2f}, {snr_stats['max']:.2f}] dB")
    
    if 'source_type' in metadata.columns:
        source_types = metadata['source_type'].value_counts()
        print(f"\nSource types:")
        for source_type, count in source_types.items():
            print(f"  {source_type}: {count}")
    
    print("="*60 + "\n")


def main():
    """
    Main execution function demonstrating dataset loading and plotting.
    """
    print("\n" + "="*60)
    print("SEISBENCH SEISMOGRAM VISUALIZATION")
    print("="*60 + "\n")
    
    try:
        # Load dataset
        print("Loading dataset...")
        dataset = load_dataset(
            data_path=DATA_DIR,
            component_order=None,  # Will use native order from dataset
            cache=None  # Set to "trace" or "full" for caching
        )
        
        # Display dataset information
        display_dataset_info(dataset)
        
        # Plot first seismogram
        print("Plotting first seismogram...")
        plot_single_seismogram(
            dataset=dataset,
            trace_idx=0,
            show_phase_arrivals=True,
            save_path="../output/seismogram_single.png"
        )
        
        # Plot multiple seismograms for comparison
        print("\nPlotting multiple seismograms...")
        n_traces = min(6, len(dataset))  # Plot up to 6 traces
        plot_multiple_seismograms(
            dataset=dataset,
            trace_indices=list(range(n_traces)),
            component=0,  # Z component
            normalize=True,
            save_path="../output/seismogram_comparison.png"
        )
        
        print("\n✓ All visualizations complete!")
        print(f"  Output directory: {Path('../output').resolve()}")
        
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please ensure metadata.csv and waveforms.hdf5 exist in the data directory.")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
