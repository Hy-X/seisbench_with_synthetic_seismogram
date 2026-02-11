# Contributing to SeisBench with Synthetic Seismogram

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/seisbench_with_synthetic_seismogram.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`

## Development Setup

### Prerequisites

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy scipy matplotlib
pip install obspy seisbench pandas h5py tqdm
pip install jupyter  # For notebook development
```

### Running Tests

```bash
# Test synthetic data generation
cd synthetic_input
python T001_test_seisbench_dataset.py

# Generate a sample seismogram
python P001_generate_synthetic_3c_seismogram.py
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use meaningful variable and function names
- Maximum line length: 100 characters (flexible for readability)

### Type Hints

Add type hints to all function signatures:

```python
def generate_wavelet(
    frequency: float,
    duration: float,
    sample_rate: int = 100
) -> np.ndarray:
    """Generate a Ricker wavelet."""
    pass
```

### Documentation

#### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def process_waveform(data: np.ndarray, filter_freq: float) -> np.ndarray:
    """Apply bandpass filter to seismic waveform.
    
    Args:
        data: Input waveform array
        filter_freq: Corner frequency in Hz
        
    Returns:
        Filtered waveform array
        
    Raises:
        ValueError: If data is empty or filter_freq is negative
    """
    pass
```

#### Comments

- Write clear, concise comments explaining **why**, not **what**
- Use inline comments sparingly
- Update comments when you update code

### File Naming Conventions

- **Scripts**: Use descriptive names with prefixes:
  - `P###_` for production scripts (e.g., `P001_generate_synthetic_3c_seismogram.py`)
  - `T###_` for test scripts (e.g., `T001_test_seisbench_dataset.py`)
  - `V###_` for visualization scripts (e.g., `V001_demo_synthetic_dataset.ipynb`)
  - `REF_` for reference notebooks (e.g., `REF_dataset_basics.ipynb`)

- **Modules**: Use lowercase with underscores (e.g., `generate_synthetic_seismogram.py`)

## Contribution Workflow

### 1. Create an Issue

Before starting work, create an issue describing:
- The problem you're solving or feature you're adding
- Your proposed approach
- Any relevant context or references

### 2. Make Your Changes

- Write clean, well-documented code
- Follow the coding standards above
- Add appropriate error handling
- Update relevant documentation

### 3. Test Your Changes

- Ensure existing tests still pass
- Add new tests for new functionality
- Test edge cases and error conditions

### 4. Update Documentation

- Update README.md if you've changed functionality
- Update docstrings and inline documentation
- Add usage examples if appropriate
- Update the project structure diagram if you've added new files

### 5. Commit Your Changes

Use clear, descriptive commit messages:

```bash
# Good commit messages
git commit -m "Add SNR calculation to batch generation script"
git commit -m "Fix: Correct P-wave polarization in vertical component"
git commit -m "Docs: Update synthetic generation configuration guide"

# Use conventional commits format
# <type>: <description>
# Types: feat, fix, docs, style, refactor, test, chore
```

### 6. Submit a Pull Request

- Push your changes to your fork
- Create a pull request with a clear title and description
- Reference any related issues
- Explain what you changed and why
- Include screenshots for UI changes

## Pull Request Checklist

Before submitting, ensure:

- [ ] Code follows PEP 8 style guidelines
- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] Tests pass successfully
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] No unnecessary files are included (check `.gitignore`)

## Areas for Contribution

We welcome contributions in these areas:

### High Priority

- **Model training pipeline**: Implement training scripts for SeisBench models
- **Evaluation metrics**: Add comprehensive evaluation and benchmarking tools
- **Data augmentation**: Implement additional augmentation strategies
- **Performance optimization**: Improve generation speed for large datasets

### Medium Priority

- **Additional noise models**: Implement more realistic noise patterns
- **Wave propagation**: Add more sophisticated wave propagation effects
- **Station geometry**: Support for network configurations
- **Real data integration**: Tools to mix synthetic with real data

### Documentation

- Tutorial notebooks for common workflows
- Video demonstrations
- API documentation
- Troubleshooting guides

## Code Review Process

1. At least one maintainer will review your PR
2. Reviewers may request changes or ask questions
3. Address feedback by pushing new commits to your branch
4. Once approved, a maintainer will merge your PR

## Getting Help

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussions
- **Email**: Contact [Hongyu Xiao](mailto:your.email@example.com) for private concerns

## Recognition

Contributors will be acknowledged in:
- README.md contributors section
- Release notes for significant contributions
- Academic papers using this code (as appropriate)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to SeisBench with Synthetic Seismogram! 🌍📊
