# Training Requirements

## Question: Does this require training using seisbench and torch?

### Short Answer
**PyTorch is required. SeisBench is optional.**

### Detailed Explanation

#### PyTorch (torch) - **REQUIRED** ✅
Yes, PyTorch is **absolutely required** for training the model:

- The U-Net model (`unet_model.py`) is implemented using PyTorch's `torch.nn` module
- The training script (`train.py`) uses PyTorch for:
  - Neural network training (`torch.nn`, `torch.optim`)
  - Data loading (`torch.utils.data.Dataset`, `DataLoader`)
  - GPU acceleration (CUDA support)
  - Model checkpointing
- All inference (predictions) requires PyTorch

**Installation:**
```bash
pip install torch>=1.10.0
```

#### SeisBench - **OPTIONAL** ⚠️
SeisBench is currently listed in `requirements.txt` but is **not actively used** in the code:

- ✅ The label generation approach is **compatible** with SeisBench's methodology
- ✅ Uses Gaussian probabilistic labels (sigma=50) similar to SeisBench
- ❌ SeisBench library itself is **not imported** anywhere in the codebase
- ❌ No direct SeisBench functionality is currently utilized

**Why is it listed?**
The repository may have originally planned to use SeisBench or includes it for:
- Future integration
- Optional data loading from SeisBench datasets
- Compatibility testing
- Reference for users familiar with SeisBench

**Can you remove it?**
Yes, if you're not using SeisBench datasets or features, you can safely remove it from `requirements.txt`.

### What You Actually Need

#### Minimal Training Setup
```bash
pip install torch>=1.10.0 numpy>=1.21.0
```

#### Full Setup (Recommended)
```bash
pip install -r requirements.txt
```

This installs:
- **torch** - Neural network framework (required)
- **numpy** - Array operations (required)
- **scipy** - Signal processing for peak detection (required for evaluation)
- **obspy** - Reading miniSEED seismic data files (required for real data)
- **matplotlib** - Visualization (required for plotting)
- **scikit-learn** - Machine learning utilities (optional)
- **h5py** - HDF5 file format support (optional)
- **seisbench** - Seismic benchmarking (optional, not currently used)

### Training Workflow

1. **Prepare your data:**
   ```python
   from data_loader import MiniSEEDLoader
   loader = MiniSEEDLoader(target_sample_rate=100)
   data, channels = loader.read_mseed('file.mseed')
   ```

2. **Create labels:**
   ```python
   from label_generator import GaussianLabelGenerator
   label_gen = GaussianLabelGenerator(sigma=50.0, sample_rate=100.0)
   labels = label_gen(n_samples=2000, picks=[5.2, 8.7])
   ```

3. **Create and train model:**
   ```python
   import torch
   from unet_model import create_unet
   from train import SeismicDataset, train_model
   
   model = create_unet(variant='standard')
   # ... create datasets and loaders ...
   history = train_model(model, train_loader, val_loader)
   ```

### Hardware Requirements

- **CPU**: Can train on CPU but will be slow
- **GPU**: Recommended for faster training
  - NVIDIA GPU with CUDA support
  - PyTorch will automatically use GPU if available

### Quick Verification

To verify your setup is correct:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch: 1.10.0 (or higher)
CUDA available: True (if you have a GPU)
```

### Summary

| Requirement | Status | Purpose |
|------------|--------|---------|
| PyTorch (torch) | **Required** | Model training, inference, GPU acceleration |
| NumPy | **Required** | Data manipulation |
| SciPy | **Required** | Peak detection in evaluation |
| ObsPy | **Required** | Reading miniSEED files |
| Matplotlib | **Required** | Visualization |
| SeisBench | **Optional** | Not currently used (compatible approach) |
| scikit-learn | **Optional** | Additional ML utilities |
| h5py | **Optional** | HDF5 file format |

---

**Last Updated:** 2026-02-10
