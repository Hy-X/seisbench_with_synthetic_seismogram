# System Architecture

## Overview

This document describes the architecture of the seismic phase picking system.

## Data Flow

```
miniSEED Files
      ↓
[Data Loader]
   • Read files
   • Resample to 100 Hz
   • Normalize data
      ↓
3-Channel Waveforms (n_channels=3, n_samples=2000)
      ↓
[U-Net Model]
   • Encoder path (4 levels)
   • Decoder path (4 levels)
   • Skip connections
      ↓
Probability Predictions (n_samples=2000)
      ↓
[Evaluation]
   • Extract peaks
   • Match with labels
   • Calculate metrics
      ↓
Performance Statistics
```

## Module Dependencies

```
example.py
    ├── data_loader.py
    ├── unet_model.py
    │   └── torch.nn
    ├── label_generator.py
    ├── evaluation.py
    │   └── scipy.signal
    └── visualization.py
        └── matplotlib

train.py
    ├── unet_model.py
    ├── label_generator.py
    └── torch.optim

batch_utils.py
    ├── data_loader.py
    ├── label_generator.py
    ├── evaluation.py
    ├── visualization.py
    └── train.py
```

## Component Details

### 1. Data Loader (`data_loader.py`)

**Purpose**: Read and preprocess miniSEED files

**Key Classes**:
- `MiniSEEDLoader`: Main loader class

**Key Methods**:
- `read_mseed()`: Read and resample data
- `create_windows()`: Create sliding windows

**Input**: miniSEED file path
**Output**: (n_channels, n_samples) numpy array

### 2. U-Net Model (`unet_model.py`)

**Purpose**: 1D U-Net architecture for phase detection

**Key Classes**:
- `UNet1D`: Main model class
- `DoubleConv`: Convolution block
- `Down`: Downsampling block
- `Up`: Upsampling block

**Architecture**:
```
Input (3, 2000)
    ↓
[Initial Conv] → filters=16
    ↓
[Down 1] → filters=32
    ↓
[Down 2] → filters=64
    ↓
[Down 3] → filters=128
    ↓
[Down 4] → filters=256 (bottleneck)
    ↓
[Up 4] → filters=128
    ↑ (skip connection)
    ↓
[Up 3] → filters=64
    ↑ (skip connection)
    ↓
[Up 2] → filters=32
    ↑ (skip connection)
    ↓
[Up 1] → filters=16
    ↑ (skip connection)
    ↓
[Output Conv] → filters=1
    ↓
Sigmoid
    ↓
Output (1, 2000)
```

### 3. Label Generator (`label_generator.py`)

**Purpose**: Generate Gaussian probabilistic labels

**Key Functions**:
- `generate_gaussian_label()`: Create label for picks
- `GaussianLabelGenerator`: Callable class

**Formula**:
```
label(x) = exp(-0.5 * ((x - pick) / sigma)²)
```

**Parameters**:
- sigma = 50 samples (0.5s at 100 Hz)
- Multiple picks: max overlap

### 4. Evaluation (`evaluation.py`)

**Purpose**: Evaluate model performance

**Key Functions**:
- `extract_picks_from_prediction()`: Find peaks
- `calculate_true_positives()`: Match predictions
- `evaluate_predictions()`: Calculate metrics

**Metrics**:
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 × Precision × Recall / (Precision + Recall)

**True Positive Criteria**:
- Predicted pick within 100 samples (1s) of true pick

### 5. Visualization (`visualization.py`)

**Purpose**: Create plots for analysis

**Key Functions**:
- `plot_waveform_with_predictions()`: Main plot
- `plot_statistics()`: Performance metrics
- `print_statistics()`: Console output

### 6. Training (`train.py`)

**Purpose**: Train the U-Net model

**Key Components**:
- `SeismicDataset`: PyTorch Dataset
- `train_model()`: Training loop
- `load_model()`: Load checkpoints

**Training Features**:
- BCE Loss
- Adam optimizer
- Learning rate scheduling
- Model checkpointing
- Validation monitoring

### 7. Batch Processing (`batch_utils.py`)

**Purpose**: Process multiple files

**Key Functions**:
- `process_mseed_directory()`: Batch read
- `batch_predict()`: Batch inference
- `save_picks_to_catalog()`: Save results
- `batch_visualize()`: Create plots

## Configuration

### Global Constants

| Constant | Value | Location |
|----------|-------|----------|
| SAMPLE_RATE | 100 Hz | All modules |
| WINDOW_LENGTH | 20 s | data_loader.py |
| N_SAMPLES | 2000 | All modules |
| SIGMA | 50 samples | label_generator.py |
| TOLERANCE | 100 samples | evaluation.py |
| THRESHOLD | 0.5 | evaluation.py, visualization.py |

### Model Variants

| Variant | base_filters | depth | kernel_size |
|---------|--------------|-------|-------------|
| small | 8 | 3 | 3 |
| standard | 16 | 4 | 3 |
| large | 32 | 4 | 5 |
| deep | 16 | 5 | 3 |

## File Organization

```
xiao_net_ver_2/
├── README.md              # Project overview
├── USAGE.md              # Usage guide
├── ARCHITECTURE.md       # This file
├── requirements.txt      # Dependencies
├── .gitignore           # Git ignore rules
│
├── data_loader.py        # Data loading
├── unet_model.py        # Model architecture
├── label_generator.py   # Label generation
├── evaluation.py        # Metrics calculation
├── visualization.py     # Plotting functions
├── train.py            # Training pipeline
├── batch_utils.py      # Batch processing
│
├── example.py          # Demo script
│
└── output/             # Generated outputs
    ├── example_1.png
    ├── example_2.png
    ├── example_3.png
    └── statistics.png
```

## Design Decisions

### 1. **Modular Design**
Each component is independent and reusable. This allows:
- Easy testing of individual components
- Flexible combination of modules
- Clear separation of concerns

### 2. **Gaussian Labels (sigma=50)**
- Provides smooth probabilistic targets
- Accounts for pick uncertainty
- Compatible with SeisBench approach
- 50 samples ≈ 0.5s spread

### 3. **1-Second Tolerance**
- Standard in seismology for automatic picking
- Balances precision and recall
- Accounts for natural pick uncertainty
- 100 samples at 100 Hz

### 4. **U-Net Architecture**
- Skip connections preserve spatial information
- Multi-scale feature extraction
- Proven effective for time series
- Flexible depth and width

### 5. **Multiple Model Variants**
- Different use cases (speed vs. accuracy)
- Flexibility for different data volumes
- Easy experimentation

## Extension Points

### Adding New Model Architectures
1. Create new model in `unet_model.py`
2. Follow same input/output interface
3. Add to `create_unet()` factory

### Supporting Different Sample Rates
1. Adjust `target_sample_rate` in MiniSEEDLoader
2. Scale `sigma` proportionally
3. Scale `tolerance_samples` proportionally

### Adding Phase Types (P, S)
1. Extend label generator for multi-output
2. Modify model `out_channels`
3. Update evaluation for per-phase metrics

### Custom Loss Functions
1. Add to `train.py`
2. Consider focal loss for imbalanced data
3. Consider dice loss for segmentation

## Performance Considerations

### Memory Usage
- Standard model: ~680K parameters ≈ 2.7 MB
- Batch size affects GPU memory
- Typical: 32-64 samples per batch

### Inference Speed
- Small model: ~1ms per window (GPU)
- Standard model: ~2ms per window (GPU)
- CPU: ~10-50ms per window

### Training Time
- Depends on dataset size
- Typical: 50-100 epochs
- ~1-2 hours for 10K samples (GPU)

## Testing Strategy

### Unit Tests (Recommended)
- Test each module independently
- Mock external dependencies
- Verify input/output shapes
- Check edge cases

### Integration Tests
- Test complete workflow
- Use synthetic data
- Verify end-to-end functionality
- Check visualization output

### Validation
- Compare with manual picks
- Use standard benchmark datasets
- Cross-validation
- Test on different stations

## Future Enhancements

### Potential Improvements
1. Multi-phase detection (P and S waves)
2. Attention mechanisms
3. Transfer learning
4. Data augmentation
5. Real-time processing
6. Uncertainty estimation
7. Multi-station processing
8. Integration with earthquake catalogs

### Performance Optimization
1. Mixed precision training
2. Model quantization
3. TorchScript compilation
4. ONNX export
5. Batch inference optimization

---

Last Updated: 2026-02-10
Version: 1.0
