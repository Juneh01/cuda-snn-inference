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

### Overview

| Feature           | Naive (V1)           | Optimized (V14)         |
| ----------------- | -------------------- | ----------------------- |
| Processing        | Sequential (1 image) | Batched (512 images)    |
| Memory Management | Alloc/Free per image | Pre-allocated Workspace |
| Data Transfer     | Synchronous          | Async + Pinned Memory   |
| Concurrency       | Single Stream        | Dual-Stream Pipeline    |
| Kernel Design     | Separate Conv/IF     | Fused Conv+IF           |
| Special Hardware  | None                 | WMMA Tensor Core        |
| Execution         | Direct Launch        | CUDA Graph              |

---

### 1️⃣ Batch Processing

**V1**: Process one image at a time with 10,000 kernel launches
**V14**: Process 512 images per batch, reducing launches to ~140

```cpp
// V1: Per-image processing
for (int i = 0; i < 10000; i++) {
    cudaMemcpy(...);  // Transfer 1 image
    kernel<<<...>>>();  // Process 1 image
}

// V14: Batch processing  
for (int batch = 0; batch < 20; batch++) {
    cudaMemcpyAsync(...);  // Transfer 512 images
    kernel<<<...>>>();  // Process 512 images in parallel
}
```

**Impact**: 🚀 **10-50x speedup** - Most critical optimization

---

### 2️⃣ Pinned Memory

**Pageable Memory (V1)**: OS can swap to disk, requires staging buffer
**Pinned Memory (V14)**: Locked in physical RAM, enables DMA transfer

```
V1 Transfer Path:
Host RAM (Pageable) → Staging Buffer (Pinned) → PCIe → GPU VRAM
                      ↑ Extra copy overhead

V14 Transfer Path:
Host RAM (Pinned) ────────────────────────────→ GPU VRAM
                   Direct DMA, no intermediate copy
```

**Impact**: 🚀 **~2x transfer speed improvement**

---

### 3️⃣ Multi-Stream Pipeline

Dual streams enable overlapping of computation and data transfer:

```
Timeline ─────────────────────────────────────────────────────▶

Stream 0: [H2D batch0][Compute batch0][D2H batch0]     [H2D batch2][Compute batch2]...
Stream 1:      [H2D batch1][Compute batch1][D2H batch1]     [H2D batch3]...
                └────── Overlapped Execution ──────┘
```

- GPU has independent Copy Engine and Compute Engine
- While Stream 0 computes, Stream 1 transfers data
- Effectively hides memory transfer latency

**Impact**: 🚀 **30-50% improvement**

---

### 4️⃣ Kernel Fusion

Fuse Conv2D and IF Neuron into single kernel to eliminate intermediate global memory access:

```
V1 (Separate Kernels):
Conv kernel: Read input → Compute → Write to Global Memory
                                            ↓
IF kernel:                    Read from Global Memory → Compute → Write result

Total: 2 Global Memory writes + 1 Global Memory read

V14 (Fused Kernel):
Fused kernel: Read input → Compute Conv → IF in registers → Write final result

Total: 1 Global Memory write (50% memory bandwidth saved)
```

**Impact**: 🚀 **15-25% improvement**

---

### 5️⃣ Shared Memory Optimization

Load weights to shared memory once, reuse across all threads in block:

```cpp
// V14: Cooperative loading to shared memory
__shared__ float s_weights[6][5][5];  // 150 floats shared by all threads
__shared__ float s_biases[6];

// All threads cooperatively load weights once
for (int i = tid; i < 150; i += blockDim.x) {
    s_weights_flat[i] = weights[i];
}
__syncthreads();

// Each thread reuses shared memory data
for (int oc = 0; oc < 6; ++oc) {
    sum = s_biases[oc];
    // Access s_weights[oc][ky][kx] - ~20 cycles vs ~400 cycles for global
}
```

| Memory Type   | Latency (cycles) | Bandwidth |
| ------------- | ---------------- | --------- |
| Registers     | 0                | Highest   |
| Shared Memory | ~20-30           | ~1.5 TB/s |
| L1 Cache      | ~30-50           | ~1 TB/s   |
| Global Memory | ~400-600         | ~900 GB/s |

**Impact**: 🚀 **20-40% improvement**

---

### 6️⃣ PTX Inline Assembly

Low-level optimizations for compute and memory operations:

```cpp
// Fused Multiply-Add: result = a * b + c in single instruction
__device__ float ptx_fma(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

// Non-coherent load: bypass L1 cache, reduce cache pollution
__device__ float ptx_ldg(const float* ptr) {
    float result;
    asm("ld.global.nc.f32 %0, [%1];" : "=f"(result) : "l"(ptr));
    return result;
}

// Vectorized load: load 4 floats (16 bytes) in single transaction
__device__ void ptx_ldg_v4(const float* ptr, float& a, float& b, float& c, float& d) {
    asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" 
        : "=f"(a), "=f"(b), "=f"(c), "=f"(d) : "l"(ptr));
}
```

| Instruction           | Description        | Benefit                                           |
| --------------------- | ------------------ | ------------------------------------------------- |
| `fma.rn.f32`          | Fused Multiply-Add | 1 instruction = 2 FLOPs, no intermediate rounding |
| `ld.global.nc.f32`    | Non-Coherent Load  | Bypass L1, reduce cache thrashing                 |
| `ld.global.nc.v4.f32` | Vectorized Load    | 4x bandwidth utilization                          |

**Impact**: 🚀 **5-10% improvement**

---

### 7️⃣ WMMA Tensor Core

Use Tensor Cores for FC layers with 16×16×16 matrix operations:

```cpp
#include <mma.h>
using namespace nvcuda;

// Declare WMMA fragments
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// Single instruction performs 16×16×16 = 4096 FMA operations
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

```
WMMA 16×16×16 Operation:
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ Input A     │   │ Weight B    │   │ Output C    │
│ 16×16 half  │ × │ 16×16 half  │ + │ 16×16 float │
└─────────────┘   └─────────────┘   └─────────────┘

CUDA Core: 4096 FMA instructions
Tensor Core: 1 mma_sync instruction → 8-16x throughput
```

**Impact**: 🚀 **2-4x speedup for FC layers**

---

### 8️⃣ Half Precision Weight Preprocessing

Pre-convert FC weights to FP16 before inference to avoid runtime conversion:

```cpp
// One-time conversion before inference
half *d_fc1_w_half, *d_fc2_w_half;
convert_weights_padded_kernel<<<...>>>(d_fc1_w, d_fc1_w_half, ...);

// During inference: directly use pre-converted half weights
wmma::load_matrix_sync(b_frag, weights_half, stride);  // No conversion overhead
```

**Impact**: 🚀 **5-10% improvement**

---

### 9️⃣ Software Pipelining with Ping-Pong Buffer

Double buffering in Conv2 kernel to overlap memory loading and computation:

```cpp
__shared__ float s_input_pingpong[2][12][12];  // Dual buffers

for (int in_c = 0; in_c < IN_C; ++in_c) {
    int curr_buf = in_c & 1;      // 0, 1, 0, 1, ...
    int next_buf = (in_c + 1) & 1; // 1, 0, 1, 0, ...
    
    // Stage 1: Prefetch next channel to alternate buffer
    if (in_c < IN_C - 1) {
        load_to_buffer(next_buf, in_c + 1);
    }
    
    // Stage 2: Compute using current buffer
    compute_convolution(curr_buf, in_c);
    
    __syncthreads();  // Ensure both stages complete
}
```

```
V1 (No Pipeline):
│Load 0│────▶│Comp 0│────▶│Load 1│────▶│Comp 1│────▶ ...
        Serial execution, Load and Compute cannot overlap

V14 (Ping-Pong Pipeline):
Buffer A: │Load 0│         │Load 2│         │Load 4│
              ╲    ╱           ╲    ╱
Compute:      │Comp 0│───│Comp 1│───│Comp 2│─── ...
              ╱    ╲           ╱    ╲  
Buffer B:    │Load 1│         │Load 3│
        Load and Compute execute in parallel
```

**Impact**: 🚀 **10-20% improvement**

---

### 🔟 CUDA Graph

Capture entire timestep loop and replay with minimal CPU overhead:

```cpp
// First execution: Capture graph
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
for (int t = 0; t < T; t++) {
    conv1_kernel<<<...>>>(...);
    pool1_kernel<<<...>>>(...);
    conv2_kernel<<<...>>>(...);
    pool2_kernel<<<...>>>(...);
    fc1_kernel<<<...>>>(...);
    fc2_kernel<<<...>>>(...);
    fc3_kernel<<<...>>>(...);
}
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&graphExec, graph, ...);

// Subsequent executions: Single launch for all 28 kernels
cudaGraphLaunch(graphExec, stream);
```

```
Traditional Launch:
CPU Overhead = 28 kernels × 5μs = 140μs per batch

CUDA Graph:
CPU Overhead = 1 launch × 5μs = 5μs per batch
Savings = 135μs per batch × 19 batches ≈ 2.5ms total
```

**Impact**: 🚀 **5-15% improvement**

---

### 1️⃣1️⃣ Workspace Pre-allocation

Allocate all GPU memory once, use pointer arithmetic for sub-buffers:

```cpp
// V1: Allocate/Free per image (10000× overhead)
for (int i = 0; i < 10000; i++) {
    cudaMalloc(&d_input, ...);   // ~100-500μs each
    cudaMalloc(&d_c1, ...);
    // ... inference ...
    cudaFree(d_input);
    cudaFree(d_c1);
}

// V14: Single allocation, pointer offsets
size_t workspace_size = BATCH_SIZE * (IN_SIZE + C1_SIZE*2 + P1_SIZE + ...);
cudaMalloc(&d_workspace, workspace_size);

float* d_ptr = d_workspace;
float* d_input = d_ptr; d_ptr += BATCH_SIZE * IN_SIZE;
float* d_c1 = d_ptr;    d_ptr += BATCH_SIZE * C1_SIZE;
// ...
```

**Impact**: 🚀 **10-20% improvement**

---

### 1️⃣2️⃣ Vectorized Memory Access

Use `float4` to load 16 bytes per transaction instead of 4 bytes:

```cpp
// Scalar load: 1 float (4 bytes) per instruction
for (int i = 0; i < 2400; i++) {
    s_weights[i] = weights[i];  // 2400 load instructions
}

// Vectorized load: 4 floats (16 bytes) per instruction
float4* s_weights_f4 = (float4*)s_weights;
const float4* weights_f4 = (const float4*)weights;
for (int i = 0; i < 600; i++) {
    s_weights_f4[i] = weights_f4[i];  // 600 load instructions (75% reduction)
}
```

**Impact**: 🚀 **5-15% improvement**

---

### 1️⃣3️⃣ Launch Bounds

Hint compiler about thread count for better register allocation:

```cpp
__global__ void __launch_bounds__(64) fused_conv_if_kernel1(...) { }
__global__ void __launch_bounds__(64) fused_conv_if_kernel2(...) { }
__global__ void __launch_bounds__(32) linear_fc3_kernel(...) { }
```

Allows compiler to allocate more registers per thread when block size is known.

---

### 📊 Optimization Summary

| Technique                | Expected Improvement | Complexity |
| ------------------------ | -------------------- | ---------- |
| Batch Processing         | 10-50×               | ⭐          |
| Pinned Memory            | 2× transfer speed    | ⭐          |
| Multi-Stream Pipeline    | 30-50%               | ⭐⭐         |
| Kernel Fusion            | 15-25%               | ⭐⭐         |
| Shared Memory            | 20-40%               | ⭐⭐         |
| PTX Intrinsics           | 5-10%                | ⭐⭐⭐        |
| WMMA Tensor Core         | 2-4× (FC layers)     | ⭐⭐⭐        |
| FP16 Weight Preprocess   | 5-10%                | ⭐⭐         |
| Software Pipelining      | 10-20%               | ⭐⭐⭐        |
| CUDA Graph               | 5-15%                | ⭐⭐         |
| Workspace Pre-allocation | 10-20%               | ⭐          |
| Vectorized Access        | 5-15%                | ⭐⭐         |

## 
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