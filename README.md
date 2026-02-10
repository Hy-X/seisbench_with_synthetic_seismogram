# xiao_net_ver_2
# XiaoNet: Lightweight Neural Network for Seismic Phase Picking and Earthquake Detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**XiaoNet** is a lightweight, edge-oriented neural network framework for **seismic phase picking**, **earthquake detection**, and **seismological signal processing**. Designed as a compact and efficient alternative to large deep-learning models such as PhaseNet and EQTransformer, this PyTorch-based framework focuses on **model compression**, **knowledge distillation**, **edge AI**, and **streaming-friendly inference** for low-power devices (e.g., Raspberry Pi 5, embedded systems, IoT devices).

## Keywords

`seismic phase picking` | `earthquake detection` | `neural network` | `deep learning` | `edge AI` | `model compression` | `knowledge distillation` | `PyTorch` | `seismology` | `U-Net` | `PhaseNet` | `EQTransformer` | `STA/LTA` | `real-time inference` | `edge computing` | `seismic waveform processing` | `earthquake early warning` | `seismological machine learning`

## Key Features

- 🪶 **Lightweight U-Net-style architectures** for seismic phase picking and earthquake detection
- 🎓 **Transfer Learning**: Fine-tuned from PhaseNet models pretrained on STEAD dataset
- 📚 **Multi-phase Support**: Handles multiple P and S phase types (P, pP, Pg, Pn, PmP, S, Sg, Sn, SmS, etc.)
- ⚡ **Edge-first design**: low memory footprint, minimal compute requirements, ultra-low latency inference
- 🔁 **Streaming-friendly inference** on continuous seismic waveforms and real-time data streams
- 🧩 **Modular codebase** for reusability, experimentation, and easy integration
- 📦 **Config-driven experiments** using JSON configuration files
- 🌐 **PyTorch-based** implementation for easy deployment and model export
- 📊 **SeisBench Integration**: Full compatibility with SeisBench datasets, models, and augmentation pipelines
- 🔬 **Research-ready** with comprehensive evaluation metrics and benchmarking tools
- 🎯 **Probabilistic Labeling**: Gaussian-smoothed phase labels for improved training stability

---

## Use Cases and Applications

**XiaoNet** is designed for various seismological and geophysical applications across operational, research, and edge computing domains.

### Research and Development

- **Seismological Machine Learning Research**: Lightweight alternative for studying neural network-based phase picking
- **Model Compression Research**: Benchmarking knowledge distillation techniques and edge AI optimization
- **Educational Applications**: Teaching tool for seismology and deep learning courses

### Edge Computing and Embedded Systems

- **Embedded Seismic Stations**: Deployment on Raspberry Pi, microcontrollers, and IoT devices
- **Field Deployments**: Low-power seismic phase picking in remote locations with limited infrastructure
- **Mobile and Portable Systems**: On-device inference for field seismology and rapid response teams

---