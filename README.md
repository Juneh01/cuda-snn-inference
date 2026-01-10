# SpikeCUDA 🚀：cuda-snn-inference
a project from GPU Architecture & Programming course, University of Chinese Academy of Sciences

## High-Performance CUDA Implementation of Spiking Neural Network Inference

[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Platform](https://img.shields.io/badge/Platform-Linux-orange.svg)]()

A highly optimized CUDA C++ implementation for Spiking Neural Network (SNN) inference on Fashion-MNIST dataset. This project achieves **~15ms inference time** for 10,000 images on NVIDIA V100 GPU, leveraging Tensor Cores, CUDA Graphs, and extensive kernel optimizations.

## 📋 Table of Contents

- [Background](#-background)
- [Features](#-features)
- [Network Architecture](#-network-architecture)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Optimization Techniques](#-optimization-techniques)
- [Performance](#-performance)

## 🎯 Background

Spiking Neural Networks (SNNs) are the third generation of neural networks that more closely mimic biological neural networks. Unlike traditional Artificial Neural Networks (ANNs), SNNs process information using discrete spikes over time, making them particularly suitable for neuromorphic computing and energy-efficient AI applications.

This project implements an optimized CUDA inference engine for a convolutional SNN trained on the Fashion-MNIST dataset. The network uses Integrate-and-Fire (IF) neurons and processes inputs over multiple timesteps (T=4).

### Key Characteristics of SNN

- **Temporal Dynamics**: Information is encoded in spike timing across T timesteps
- **Binary Activation**: Neurons output binary spikes (0 or 1) based on membrane potential
- **Membrane Potential**: Accumulates over time and resets after firing
- **Event-Driven**: Sparse, energy-efficient computation

## ✨ Features

- **High Performance**: ~15ms for 10,000 images on V100
- **Tensor Core Acceleration**: WMMA (Warp Matrix Multiply-Accumulate) for FC layers
- **CUDA Graph**: Reduced kernel launch overhead
- **PTX Intrinsics**: Low-level optimizations for memory access
- **Multi-Stream Pipeline**: Overlapped computation and data transfer
- **FP16 Optimization**: Half-precision weights for Tensor Core utilization

## 🧠 Network Architecture
### Side-by-side comparison between naive and optimized version
```
┌──────────────────────────────────┐    ┌──────────────────────────────────┐
│         NAIVE VERSION            │    │       OPTIMIZED VERSION          │
│      (Sequential, 12 Kernels)    │    │   (Batched, 7 Fused Kernels)     │
├──────────────────────────────────┤    ├──────────────────────────────────┤
│                                  │    │                                  │
│   Input (1 × 28 × 28)            │    │   Input (512 × 28 × 28)          │
│            │                     │    │            │                     │
│            ▼                     │    │            ▼                     │
│   ┌─────────────┐                │    │   ┌─────────────────────┐        │
│   │   Conv2D    │ K1             │    │   │ Conv2D + IF (Fused) │ K1     │
│   └──────┬──────┘                │    │   │  + Shared Memory    │        │
│          ▼                       │    │   │  + PTX Intrinsics   │        │
│   ┌─────────────┐                │    │   └─────────┬───────────┘        │
│   │  IF Neuron  │ K2             │    │             │                    │
│   └──────┬──────┘                │    │             ▼                    │
│          ▼                       │    │   ┌─────────────────────┐        │
│   ┌─────────────┐                │    │   │     MaxPool 2×2     │ K2     │
│   │  MaxPool    │ K3             │    │   └─────────┬───────────┘        │
│   └──────┬──────┘                │    │             │                    │
│          ▼                       │    │             ▼                    │
│   ┌─────────────┐                │    │   ┌─────────────────────┐        │
│   │   Conv2D    │ K4             │    │   │ Conv2D + IF (Fused) │ K3     │
│   └──────┬──────┘                │    │   │  + Ping-Pong Buffer │        │
│          ▼                       │    │   │  + Software Pipeline│        │
│   ┌─────────────┐                │    │   └─────────┬───────────┘        │
│   │  IF Neuron  │ K5             │    │             │                    │
│   └──────┬──────┘                │    │             ▼                    │
│          ▼                       │    │   ┌─────────────────────┐        │
│   ┌─────────────┐                │    │   │     MaxPool 2×2     │ K4     │
│   │  MaxPool    │ K6             │    │   └─────────┬───────────┘        │
│   └──────┬──────┘                │    │             │                    │
│          ▼                       │    │             ▼                    │
│   ┌─────────────┐                │    │   ┌─────────────────────┐        │
│   │     FC1     │ K7             │    │   │  FC1 + IF (WMMA)    │ K5     │
│   └──────┬──────┘                │    │   │  Tensor Core FP16   │        │
│          ▼                       │    │   └─────────┬───────────┘        │
│   ┌─────────────┐                │    │             │                    │
│   │  IF Neuron  │ K8             │    │             ▼                    │
│   └──────┬──────┘                │    │   ┌─────────────────────┐        │
│          ▼                       │    │   │  FC2 + IF (WMMA)    │ K6     │
│   ┌─────────────┐                │    │   │  Tensor Core FP16   │        │
│   │     FC2     │ K9             │    │   └─────────┬───────────┘        │
│   └──────┬──────┘                │    │             │                    │
│          ▼                       │    │             ▼                    │
│   ┌─────────────┐                │    │   ┌─────────────────────┐        │
│   │  IF Neuron  │ K10            │    │   │ FC3 + Accumulate    │ K7     │
│   └──────┬──────┘                │    │   │  Float4 Vectorized  │        │
│          ▼                       │    │   └─────────┬───────────┘        │
│   ┌─────────────┐                │    │             │                    │
│   │     FC3     │ K11            │    │             ▼                    │
│   └──────┬──────┘                │    │      Output (512 × 10)           │
│          ▼                       │    │                                  │
│   ┌─────────────┐                │    ├──────────────────────────────────┤
│   │ Accumulate  │ K12            │    │  Additional Optimizations:       │
│   └──────┬──────┘                │    │  • CUDA Graph                    │
│          ▼                       │    │  • Multi-Stream Pipeline         │
│      Output (10)                 │    │  • Pinned Memory                 │
│                                  │    │  • Async Transfers               │
├──────────────────────────────────┤    ├──────────────────────────────────┤
│  Kernels: 12                     │    │  Kernels: 7                      │
│  Batch Size: 1                   │    │  Batch Size: 512                 │
│  Memory Access: Naive            │    │  Memory Access: Optimized        │
│  Compute: FP32 only              │    │  Compute: FP32 + FP16 Tensor     │
│  Timesteps: 8                    │    │  Timesteps: 4                    │
└──────────────────────────────────┘    └──────────────────────────────────┘
```

**Timesteps**: T=4 (network processes each input over 4 time steps)

## 💻 Requirements

### Hardware

- NVIDIA GPU with Compute Capability ≥ 7.0 (Volta or newer)
- Recommended: Tesla V100, RTX 2080 or newer

### Software

- CUDA Toolkit 11.8
- GCC/G++ with C++14 support
- Linux (Ubuntu 20.04+ recommended)

### For Training (Optional)

- Python 3.12
- PyTorch 2.6.0
- SpikingJelly


## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Juneh01/cuda-snn-inference.git
cd cuda-snn-inference
```

### 2. Download Dataset

Download Fashion-MNIST dataset and place it in the `data` directory:

```bash
mkdir -p data/FashionMNIST/raw
cd data/FashionMNIST/raw
# Download the following files:
# - t10k-images-idx3-ubyte
# - t10k-labels-idx1-ubyte
```


### 3. Prepare Weights

Use the pre-trained weights provided in `weights/` directory, or train your own:

```bash
# Using pre-trained weights
cp weights/*.txt ./here/are/weights

# Or train your own (requires Python environment)
python train.py
```

### 4. Compile

On the course evaluation system (V100)
```bash
nvcc inference_optimized.cu -o inference_optimized_prog \
    -Xcompiler "-O3 -std=c++14" \
    -gencode arch=compute_70,code=sm_70 \
    -rdc=true
```

For different GPU architectures or local evaluation:

```bash
# For Ada Lovelace (my RTX 4070 Super)
nvcc inference_optimized.cu -o ./inference_optimized_prog \
    -Xcompiler "-O3 -std=c++17" \
    -gencode arch=compute_89,code=sm_89 \
    -rdc=true

# For multiple architectures
nvcc inference_optimized.cu -o inference_optimized_prog \
    -Xcompiler "-O3 -std=c++14" \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_75,code=sm_75 \
    -gencode arch=compute_80,code=sm_80 \
    -rdc=true
```

## 🚀 Usage

### Basic Usage

```bash
./inference_prog <path_to_weights_directory>
```

### Example

```bash
./inference_optimized_prog ./here/are/weights/
```

### Output Format

```
<inference_time>:<accuracy>
```

Example output:

```
0.0154:0.8989
```

- Inference time: 0.0154 seconds (15.4 ms)
- Accuracy: 89.89%



## 📁 Project Structure

```
cuda-snn-inference/
├── README.md                 # This file
├── LICENSE                   # MIT License
├── inference.cu              # Main CUDA inference implementation
├── here/are/weights/                  # Pre-trained model weights
│   ├── conv1.weight.txt
│   ├── conv1.bias.txt
│   ├── conv2.weight.txt
│   ├── conv2.bias.txt
│   ├── fc1.weight.txt
│   ├── fc1.bias.txt
│   ├── fc2.weight.txt
│   ├── fc2.bias.txt
│   ├── fc3.weight.txt
│   └── fc3.bias.txt
├── data/                     # Dataset directory
│   └── FashionMNIST/
│       └── raw/
│           ├── t10k-images-idx3-ubyte
│           └── t10k-labels-idx1-ubyte
└── train.py                 # Training scripts (Python)
```

## ⚡ Optimization Techniques

### 1. Tensor Core (WMMA)

- FC1 and FC2 layers use Warp Matrix Multiply-Accumulate
- FP16 weights pre-converted for Tensor Core efficiency
- 16×16×16 tile size for optimal utilization

### 2. CUDA Graph

- Captures entire timestep loop (28 kernel launches)
- Eliminates CPU-GPU synchronization overhead
- ~1-2ms savings per batch

### 3. PTX Intrinsics

```cuda
// Fused Multiply-Add
asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));

// Non-coherent cache load
asm("ld.global.nc.f32 %0, [%1];" : "=f"(result) : "l"(ptr));

// Vectorized load (float4)
asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" ...);
```

### 4. Memory Optimization

- Shared memory for weights and intermediate results
- Pinned host memory for async transfers
- Coalesced global memory access patterns

### 5. Multi-Stream Pipeline

- 2 CUDA streams for overlapped execution
- Async memory transfers (H2D and D2H)
- Double buffering for continuous processing

### 6. Kernel Fusion

- Conv + IF neuron fused into single kernel
- Reduced global memory traffic
- Software pipelining with ping-pong buffers

## 📊 Performance

### Benchmark Results (V100-PCIE-32GB)

| Metric         | Value               |
| -------------- | ------------------- |
| Total Images   | 10,000              |
| Batch Size     | 512                 |
| Timesteps (T)  | 4                   |
| Inference Time | ~15.4 ms            |
| Throughput     | ~645,000 images/sec |
| Accuracy       | 89.89%              |

### Performance Breakdown

| Component           | Estimated Time |
| ------------------- | -------------- |
| Data Transfer (H2D) | ~2 ms          |
| Conv1 + IF          | ~3 ms          |
| Pool1               | ~0.5 ms        |
| Conv2 + IF          | ~2 ms          |
| Pool2               | ~0.3 ms        |
| FC1 (WMMA)          | ~2 ms          |
| FC2 (WMMA)          | ~1.5 ms        |
| FC3                 | ~1 ms          |
| Data Transfer (D2H) | ~0.5 ms        |
| Overhead            | ~2 ms          |

### Comparison

| Implementation           | Time (ms) | Speedup |
| ------------------------ | --------- | ------- |
| Naive CUDA               | ~4903     | 1×      |
| Optimized (this project) | ~15.4     | ~318×   |
| Theoretical Limit*       | ~4        | -       |

*Based on memory bandwidth analysis

## 🔬 Training (Optional)

### Setup Python Environment

```bash
conda create -n snn-cuda python=3.12
conda activate snn-cuda
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
pip install spikingjelly
```

### Train Model

```bash
python train.py
```

Training parameters can be modified in `train.py`:

- Epochs: 100
- Batch size: 128
- Learning rate: 1e-3
- Timesteps: 4



## 🙏 Acknowledgments

- Course: GPU Architecture and Programming (2025Fall) UCAS
- Framework: [SpikingJelly](https://github.com/fangwei123456/spikingjelly)
- Dataset: [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

## 📧 Contact

For questions or suggestions, please open an issue or contact the maintainer.

---

**Made with ❤️ and CUDA**