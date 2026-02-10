"""
Plot seismograms by directly loading from HDF5 and CSV files.

This script provides a lightweight alternative to using SeisBench's dataset
loader, directly accessing waveform data from HDF5 and metadata from CSV.

Usage:
    python plot_seismograms_direct.py
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Union
import warnings


# Configuration constants
DATA_DIR = Path(__file__).parent.parent / "data"
METADATA_FILE = DATA_DIR / "metadata.csv"
WAVEFORMS_FILE = DATA_DIR / "waveforms.hdf5"
SAMPLE_RATE = 100.0  # Hz
FIGURE_DPI = 100


def load_metadata() -> pd.DataFrame:
    """
    Load seismogram metadata from CSV file.
    
    Returns:
        DataFrame containing trace metadata
        
    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    
    df = pd.read_csv(METADATA_FILE)
    print(f"✓ Loaded metadata: {len(df)} traces")
    return df


def load_waveform(
    trace_name: str,
    hdf5_file: Union[str, Path, h5py.File]
) -> np.ndarray:
    """
    Load waveform data for a specific trace from HDF5 file.
    
    Args:
        trace_name: Name of the trace to load
        hdf5_file: Path to HDF5 file or open h5py.File object
        
    Returns:
        Waveform array of shape (3, n_samples) for Z, N, E components
        
    Raises:
        KeyError: If trace_name not found in HDF5 file
        ValueError: If waveform data has incorrect shape
    """
    close_file = False
    
    if isinstance(hdf5_file, (str, Path)):
        hdf5_file = h5py.File(hdf5_file, 'r')
        close_file = True
    
    try:
        if trace_name not in hdf5_file:
            raise KeyError(f"Trace '{trace_name}' not found in HDF5 file")
        
        trace_group = hdf5_file[trace_name]
        
        # Handle both group structure (with 'data' dataset) and direct dataset
        if isinstance(trace_group, h5py.Group) and 'data' in trace_group:
            waveform = trace_group['data'][:]
        elif isinstance(trace_group, h5py.Dataset):
            waveform = trace_group[:]
        else:
            raise ValueError(f"Unexpected HDF5 structure for trace '{trace_name}'")
        
        if waveform.ndim != 2 or waveform.shape[0] != 3:
            raise ValueError(
                f"Expected waveform shape (3, n_samples), got {waveform.shape}"
            )
        
        return waveform
        
    finally:
        if close_file:
            hdf5_file.close()


def plot_single_seismogram(
    metadata_df: pd.DataFrame,
    trace_idx: int,
    hdf5_file: Union[str, Path],
    show_phase_arrivals: bool = True,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    Plot a single three-component seismogram with optional phase arrivals.
    
    Creates a multi-panel plot showing Z, N, and E components with
    P and S wave arrival markers if available in metadata.
    
    Args:
        metadata_df: DataFrame containing trace metadata
        trace_idx: Index into metadata_df
        hdf5_file: Path to HDF5 waveforms file
        show_phase_arrivals: Whether to mark P and S wave arrivals
        figsize: Figure size as (width, height) in inches
        save_path: Optional path to save figure. If None, displays interactively
        
    Returns:
        Matplotlib figure object
        
    Raises:
        IndexError: If trace_idx is out of range
        ValueError: If waveform data is invalid
    """
    if trace_idx < 0 or trace_idx >= len(metadata_df):
        raise IndexError(
            f"Trace index {trace_idx} out of range [0, {len(metadata_df)-1}]"
        )
    
    # Get metadata
    metadata = metadata_df.iloc[trace_idx]
    trace_name = metadata['trace_name']
    
    # Load waveform
    waveforms = load_waveform(trace_name, hdf5_file)
    
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
    metadata_df: pd.DataFrame,
    trace_indices: List[int],
    hdf5_file: Union[str, Path],
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
        metadata_df: DataFrame containing trace metadata
        trace_indices: List of indices into metadata_df
        hdf5_file: Path to HDF5 waveforms file
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
            # Get metadata and load waveform
            metadata = metadata_df.iloc[trace_idx]
            trace_name = metadata['trace_name']
            waveforms = load_waveform(trace_name, hdf5_file)
            
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


def display_dataset_info(metadata_df: pd.DataFrame) -> None:
    """
    Display summary information about the dataset.
    
    Args:
        metadata_df: DataFrame containing trace metadata
    """
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Total traces: {len(metadata_df)}")
    print(f"Metadata columns: {list(metadata_df.columns)}")
    
    # Get statistics
    if 'trace_sampling_rate_hz' in metadata_df.columns:
        sample_rates = metadata_df['trace_sampling_rate_hz'].unique()
        print(f"Sampling rate(s): {sample_rates} Hz")
    
    if 'trace_npts' in metadata_df.columns:
        n_samples = metadata_df['trace_npts'].unique()
        durations = n_samples / SAMPLE_RATE
        print(f"Trace duration(s): {durations} seconds")
    
    if 'snr_db' in metadata_df.columns:
        snr_stats = metadata_df['snr_db'].describe()
        print(f"\nSNR statistics:")
        print(f"  Mean: {snr_stats['mean']:.2f} dB")
        print(f"  Median: {snr_stats['50%']:.2f} dB")
        print(f"  Range: [{snr_stats['min']:.2f}, {snr_stats['max']:.2f}] dB")
    
    if 'source_type' in metadata_df.columns:
        source_types = metadata_df['source_type'].value_counts()
        print(f"\nSource types:")
        for source_type, count in source_types.items():
            print(f"  {source_type}: {count}")
    
    print("="*60 + "\n")


def main():
    """
    Main execution function demonstrating seismogram loading and plotting.
    """
    print("\n" + "="*60)
    print("SEISMOGRAM VISUALIZATION (Direct Loading)")
    print("="*60 + "\n")
    
    try:
        # Check for required files
        if not WAVEFORMS_FILE.exists():
            raise FileNotFoundError(f"Waveforms file not found: {WAVEFORMS_FILE}")
        
        # Load metadata
        print("Loading metadata...")
        metadata = load_metadata()
        
        # Display dataset information
        display_dataset_info(metadata)
        
        # Plot first seismogram
        print("Plotting first seismogram...")
        plot_single_seismogram(
            metadata_df=metadata,
            trace_idx=0,
            hdf5_file=WAVEFORMS_FILE,
            show_phase_arrivals=True,
            save_path="../output/seismogram_single.png"
        )
        
        # Plot multiple seismograms for comparison
        print("\nPlotting multiple seismograms...")
        n_traces = min(6, len(metadata))  # Plot up to 6 traces
        plot_multiple_seismograms(
            metadata_df=metadata,
            trace_indices=list(range(n_traces)),
            hdf5_file=WAVEFORMS_FILE,
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
