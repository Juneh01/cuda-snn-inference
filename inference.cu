#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY BEGIN
// ===================================================================================
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}
// ===================================================================================
// Helper for CUDA Error Handling - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY BEGIN
// ===================================================================================
std::vector<std::vector<float>> read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_images, 4); num_images = __builtin_bswap32(num_images);
    file.read((char*)&num_rows, 4); num_rows = __builtin_bswap32(num_rows);
    file.read((char*)&num_cols, 4); num_cols = __builtin_bswap32(num_cols);
    std::vector<std::vector<float>> images(num_images, std::vector<float>(num_rows * num_cols));
    std::vector<unsigned char> buffer(num_rows * num_cols);
    for (int i = 0; i < num_images; ++i) {
        file.read((char*)buffer.data(), buffer.size());
        for (size_t j = 0; j < buffer.size(); ++j) {
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f;
        }
    }
    return images;
}

std::vector<int> read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { std::cerr << "Cannot open file: " << path << std::endl; return {}; }
    int magic_number = 0, num_items = 0;
    file.read((char*)&magic_number, 4); magic_number = __builtin_bswap32(magic_number);
    file.read((char*)&num_items, 4); num_items = __builtin_bswap32(num_items);
    std::vector<int> labels(num_items);
    std::vector<unsigned char> buffer(num_items);
    file.read((char*)buffer.data(), num_items);
    for(int i = 0; i < num_items; ++i) { labels[i] = static_cast<int>(buffer[i]); }
    return labels;
}

std::vector<float> read_param(const std::string& path) {
    std::ifstream file(path);
    if (!file) { std::cerr << "Cannot open parameter file: " << path << std::endl; return {}; }
    std::vector<float> params; float param;
    while (file >> param) { params.push_back(param); }
    return params;
}
// ===================================================================================
// Data and Parameter Loading Functions - DO NOT MODIFY END
// ===================================================================================

// ===================================================================================
// PTX Intrinsics
// ===================================================================================
__device__ __forceinline__ float ptx_fma(float a, float b, float c) {
    float result;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(result) : "f"(a), "f"(b), "f"(c));
    return result;
}

__device__ __forceinline__ float ptx_ldg(const float* ptr) {
    float result;
    asm("ld.global.nc.f32 %0, [%1];" : "=f"(result) : "l"(ptr));
    return result;
}

__device__ __forceinline__ void ptx_ldg_v4(const float* ptr, float& a, float& b, float& c, float& d) {
    asm("ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];" 
        : "=f"(a), "=f"(b), "=f"(c), "=f"(d) 
        : "l"(ptr));
}

// ===================================================================================
// Kernel: Conv1 + IF (Keep V10 implementation, it's efficient for small scale)
// ===================================================================================
__global__ void __launch_bounds__(64) fused_conv_if_kernel1_ptx(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ membrane_potential,
    float* __restrict__ spikes,
    int batch_size) {
    
    const int IN_H = 28, IN_W = 28;
    const int OUT_C = 6, OUT_H = 24, OUT_W = 24;
    const int K = 5;
    
    __shared__ float s_weights[OUT_C][K][K];
    __shared__ float s_biases[OUT_C];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int threads_per_block = blockDim.x * blockDim.y;
    int b = blockIdx.z;
    
    for (int i = tid; i < OUT_C * K * K; i += threads_per_block) {
        ((float*)s_weights)[i] = ptx_ldg(&weights[i]);
    }
    if (tid < OUT_C) {
        s_biases[tid] = ptx_ldg(&biases[tid]);
    }
    __syncthreads();
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b < batch_size && out_y < OUT_H && out_x < OUT_W) {
        const float* input_ptr = input + b * IN_H * IN_W;
        
        #pragma unroll
        for (int oc = 0; oc < OUT_C; ++oc) {
            float sum = s_biases[oc];
            #pragma unroll
            for (int ky = 0; ky < K; ++ky) {
                int in_row = (out_y + ky) * IN_W + out_x;
                #pragma unroll
                for (int kx = 0; kx < K; ++kx) {
                    float in_val = ptx_ldg(&input_ptr[in_row + kx]);
                    sum = ptx_fma(in_val, s_weights[oc][ky][kx], sum);
                }
            }
            int out_idx = b * OUT_C * OUT_H * OUT_W + oc * OUT_H * OUT_W + out_y * OUT_W + out_x;
            float v = membrane_potential[out_idx] + sum;
            float spike = (v >= 1.0f) ? 1.0f : 0.0f;
            spikes[out_idx] = spike;
            membrane_potential[out_idx] = v * (1.0f - spike);
        }
    }
}

// ===================================================================================
// Kernel: Conv2 + IF (Software Pipelining with Ping-Pong Buffer)
// 参考优化帖的双缓冲流水线设计
// ===================================================================================
__global__ void __launch_bounds__(64) fused_conv_if_kernel2_ptx(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ membrane_potential,
    float* __restrict__ spikes,
    int batch_size) {
    
    const int IN_C = 6, IN_H = 12, IN_W = 12;
    const int OUT_C = 16, OUT_H = 8, OUT_W = 8;
    const int K = 5;
    const int SLICE_SIZE = IN_H * IN_W;
    const int SLICE_SIZE_F4 = SLICE_SIZE / 4; // 36 float4s
    
    // ===== 共享内存声明 =====
    // 1. 输入数据双缓冲 (Ping-Pong Buffer) 用于软件流水线
    __shared__ float s_input_pingpong[2][IN_H][IN_W];
    // 2. 权重和偏置一次性加载
    __shared__ float s_weights[OUT_C][IN_C][K][K];
    __shared__ float s_biases[OUT_C];
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads_per_block = blockDim.x * blockDim.y; // 64
    int b = blockIdx.z;
    
    // ===== 1. 初始数据加载 (非流水线部分) =====
    // 加载全部权重 (2400 floats = 600 float4s)
    float4* s_weights_f4 = (float4*)&s_weights[0][0][0][0];
    const float4* global_weights_f4 = (const float4*)weights;
    #pragma unroll
    for (int i = tid; i < 600; i += threads_per_block) {
        s_weights_f4[i] = global_weights_f4[i];
    }
    
    // 加载偏置
    if (tid < OUT_C) {
        s_biases[tid] = ptx_ldg(&biases[tid]);
    }
    
    // 寄存器累加器
    float accum[OUT_C];
    #pragma unroll
    for (int i = 0; i < OUT_C; ++i) accum[i] = 0.0f;
    
    // ===== 2. 软件流水线 (Software Pipelining) =====
    const float4* global_input_base_f4 = (const float4*)(input + (long long)b * IN_C * IN_H * IN_W);
    
    // --- 流水线启动 (Prologue) ---
    // 预加载第一个输入切片 (in_c = 0) 到缓冲区 0
    float4* s_input_slice0_f4 = (float4*)&s_input_pingpong[0][0][0];
    #pragma unroll
    for (int i = tid; i < SLICE_SIZE_F4; i += threads_per_block) {
        s_input_slice0_f4[i] = global_input_base_f4[i];
    }
    __syncthreads();
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // --- 流水线主循环 (Steady State) ---
    #pragma unroll 1
    for (int in_c = 0; in_c < IN_C; ++in_c) {
        int current_buf = in_c & 1;
        int next_buf = (in_c + 1) & 1;
        
        // --- STAGE 1: 异步预取下一个输入切片 ---
        if (in_c < IN_C - 1) {
            const float4* global_next_input_f4 = global_input_base_f4 + (in_c + 1) * SLICE_SIZE_F4;
            float4* s_input_next_f4 = (float4*)&s_input_pingpong[next_buf][0][0];
            #pragma unroll
            for (int i = tid; i < SLICE_SIZE_F4; i += threads_per_block) {
                s_input_next_f4[i] = global_next_input_f4[i];
            }
        }
        
        // --- STAGE 2: 计算 (使用当前缓冲区) ---
        if (out_y < OUT_H && out_x < OUT_W) {
            #pragma unroll
            for (int ky = 0; ky < K; ++ky) {
                #pragma unroll
                for (int kx = 0; kx < K; ++kx) {
                    float in_val = s_input_pingpong[current_buf][out_y + ky][out_x + kx];
                    #pragma unroll
                    for (int oc = 0; oc < OUT_C; ++oc) {
                        accum[oc] = ptx_fma(in_val, s_weights[oc][in_c][ky][kx], accum[oc]);
                    }
                }
            }
        }
        
        // --- STAGE 3: 同步 ---
        __syncthreads();
    }
    
    // ===== 3. 计算收尾与写回 =====
    if (b < batch_size && out_y < OUT_H && out_x < OUT_W) {
        long long output_base_idx = (long long)b * OUT_C * OUT_H * OUT_W + out_y * OUT_W + out_x;
        
        #pragma unroll
        for (int oc = 0; oc < OUT_C; ++oc) {
            long long out_idx = output_base_idx + (long long)oc * OUT_H * OUT_W;
            float v = membrane_potential[out_idx] + accum[oc] + s_biases[oc];
            float spike = (v >= 1.0f) ? 1.0f : 0.0f;
            spikes[out_idx] = spike;
            membrane_potential[out_idx] = v * (1.0f - spike);
        }
    }
}

// ===================================================================================
// Kernel: MaxPool (Standard V10)
// ===================================================================================
__global__ void maxpool2d_kernel_ptx(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w) {
    
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int bc = blockIdx.z;
    int b = bc / channels;
    int c = bc % channels;
    
    if (b < batch_size && out_y < out_h && out_x < out_w) {
        int in_y = out_y * 2;
        int in_x = out_x * 2;
        const float* in_ptr = input + b * channels * in_h * in_w + c * in_h * in_w;
        float v0 = ptx_ldg(&in_ptr[in_y * in_w + in_x]);
        float v1 = ptx_ldg(&in_ptr[in_y * in_w + in_x + 1]);
        float v2 = ptx_ldg(&in_ptr[(in_y + 1) * in_w + in_x]);
        float v3 = ptx_ldg(&in_ptr[(in_y + 1) * in_w + in_x + 1]);
        output[b * channels * out_h * out_w + c * out_h * out_w + out_y * out_w + out_x] = fmaxf(fmaxf(v0, v1), fmaxf(v2, v3));
    }
}

// ===================================================================================
// Kernel: Convert float weights to half (preprocessing)
// ===================================================================================
__global__ void convert_weights_to_half_kernel(
    const float* __restrict__ weights_float,
    half* __restrict__ weights_half,
    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights_half[idx] = __float2half(weights_float[idx]);
    }
}

// ===================================================================================
// Kernel: WMMA Linear Layer with Pre-converted Half Weights (FC1, FC2)
// Weights are already in half precision - no runtime conversion needed
// Grid: (OUT_F / 16, BATCH_SIZE / 16)
// Block: 32 threads (1 Warp)
// ===================================================================================
template <int IN_F, int OUT_F, int IN_F_PADDED>
__global__ void wmma_linear_if_kernel_half(
    const float* __restrict__ input,
    const half* __restrict__ weights_half,  // Pre-converted half weights
    const float* __restrict__ biases,
    float* __restrict__ membrane_potential,
    float* __restrict__ spikes,
    int batch_size) {
    
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    int global_warp_m = blockIdx.y; // Batch dim
    int global_warp_n = blockIdx.x; // Output feature dim
    
    // Only need shared memory for input (float->half conversion)
    // Weights are already half in global memory
    __shared__ half s_a[WMMA_M * WMMA_K];
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    wmma::fill_fragment(c_frag, 0.0f);
    
    int tid = threadIdx.x;
    
    // Loop over K dimension
    for (int k = 0; k < IN_F_PADDED; k += WMMA_K) {
        
        // Load Input A tile and convert to half
        #pragma unroll
        for (int i = tid; i < WMMA_M * WMMA_K; i += 32) {
            int r = i / WMMA_K;
            int c = i % WMMA_K;
            int g_row = global_warp_m * WMMA_M + r;
            int g_col = k + c;
            
            float val = 0.0f;
            if (g_row < batch_size && g_col < IN_F) {
                val = input[g_row * IN_F + g_col];
            }
            s_a[i] = __float2half(val);
        }
        
        __syncwarp();
        
        // Load fragments - weights directly from global (already half)
        wmma::load_matrix_sync(a_frag, s_a, WMMA_K);
        
        // Weight layout: [OUT_F_PADDED][IN_F_PADDED], stored for col_major access
        // Each 16x16 tile at position (global_warp_n, k/16)
        const half* w_tile = weights_half + global_warp_n * WMMA_N * IN_F_PADDED + k;
        wmma::load_matrix_sync(b_frag, w_tile, IN_F_PADDED);
        
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        
        __syncwarp();
    }
    
    // Store and post-process
    __shared__ float s_c[WMMA_M * WMMA_N];
    wmma::store_matrix_sync(s_c, c_frag, WMMA_N, wmma::mem_row_major);
    __syncwarp();
    
    #pragma unroll
    for (int i = tid; i < WMMA_M * WMMA_N; i += 32) {
        int r = i / WMMA_N;
        int c = i % WMMA_N;
        
        int g_batch = global_warp_m * WMMA_M + r;
        int g_out = global_warp_n * WMMA_N + c;
        
        if (g_batch < batch_size && g_out < OUT_F) {
            float sum = s_c[i] + biases[g_out];
            int idx = g_batch * OUT_F + g_out;
            
            float v = membrane_potential[idx] + sum;
            float spike = (v >= 1.0f) ? 1.0f : 0.0f;
            spikes[idx] = spike;
            membrane_potential[idx] = v * (1.0f - spike);
        }
    }
}

// Kernel for padded weight conversion
__global__ void convert_weights_padded_kernel(
    const float* __restrict__ weights_float,
    half* __restrict__ weights_half,
    int out_f, int in_f,
    int in_f_padded) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = out_f * in_f;
    if (idx < total) {
        int out_idx = idx / in_f;
        int in_idx = idx % in_f;
        weights_half[out_idx * in_f_padded + in_idx] = __float2half(weights_float[idx]);
    }
}

// ===================================================================================
// Kernel: Optimized Linear Accumulate (FC3: 84->10)
// Kept non-WMMA because N=10 < 16, padding overhead is high.
// Optimized with float4 weights loading.
// ===================================================================================
__global__ void __launch_bounds__(32) linear_accumulate_fc3_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ output,
    int batch_size) {
    
    const int IN_F = 84, OUT_F = 10;
    __shared__ float s_input[IN_F];
    
    int b = blockIdx.y;
    if (b >= batch_size) return;
    
    // Collaborative load input
    const float* input_row = input + b * IN_F;
    for (int i = threadIdx.x; i < IN_F; i += blockDim.x) {
        s_input[i] = ptx_ldg(&input_row[i]);
    }
    __syncthreads();
    
    int out_idx = threadIdx.x;
    if (out_idx < OUT_F) {
        float sum = ptx_ldg(&biases[out_idx]);
        const float* w_ptr = weights + out_idx * IN_F;
        
        // Manually unrolled loop for 84 elements (21 float4s)
        #pragma unroll
        for (int i = 0; i < 84; i += 4) {
            float4 w;
            ptx_ldg_v4(&w_ptr[i], w.x, w.y, w.z, w.w);
            sum = ptx_fma(s_input[i], w.x, sum);
            sum = ptx_fma(s_input[i+1], w.y, sum);
            sum = ptx_fma(s_input[i+2], w.z, sum);
            sum = ptx_fma(s_input[i+3], w.w, sum);
        }
        
        output[b * OUT_F + out_idx] += sum;
    }
}


// ===================================================================================
// INFERENCE FUNCTION with Half Weights Preprocessing + CUDA Graph
// ===================================================================================

std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    float *d_conv1_w, float *d_conv1_b, float *d_conv2_w, float *d_conv2_b,
    float *d_fc1_w, float *d_fc1_b, float *d_fc2_w, float *d_fc2_b, float *d_fc3_w, float *d_fc3_b) {
    
    const int total_images = images.size();
    const int BATCH_SIZE = 512; 
    const int num_batches = (total_images + BATCH_SIZE - 1) / BATCH_SIZE;
    const int T = 4;
    
    const int IN_SIZE = 28 * 28;
    const int C1_OUT = 6, C1_H = 24, C1_W = 24, C1_SIZE = C1_OUT * C1_H * C1_W;
    const int P1_H = 12, P1_W = 12, P1_SIZE = C1_OUT * P1_H * P1_W;
    const int C2_OUT = 16, C2_H = 8, C2_W = 8, C2_SIZE = C2_OUT * C2_H * C2_W;
    const int P2_H = 4, P2_W = 4, P2_SIZE = C2_OUT * P2_H * P2_W;
    const int FC1_IN = 256, FC1_OUT = 120;
    const int FC2_IN = 120, FC2_OUT = 84;
    const int FC3_OUT = 10;
    
    // Padded dimensions for WMMA (must be multiple of 16)
    const int FC1_IN_PADDED = 256;   // Already aligned
    const int FC1_OUT_PADDED = 128;  // 120 -> 128
    const int FC2_IN_PADDED = 128;   // 120 -> 128
    const int FC2_OUT_PADDED = 96;   // 84 -> 96
    
    // =========================================================================
    // 1. Preprocess FC weights to Half precision with proper padding
    // =========================================================================
    half *d_fc1_w_half, *d_fc2_w_half;
    checkCudaErrors(cudaMalloc(&d_fc1_w_half, FC1_OUT_PADDED * FC1_IN_PADDED * sizeof(half)));
    checkCudaErrors(cudaMalloc(&d_fc2_w_half, FC2_OUT_PADDED * FC2_IN_PADDED * sizeof(half)));
    
    // Zero out padded buffers
    checkCudaErrors(cudaMemset(d_fc1_w_half, 0, FC1_OUT_PADDED * FC1_IN_PADDED * sizeof(half)));
    checkCudaErrors(cudaMemset(d_fc2_w_half, 0, FC2_OUT_PADDED * FC2_IN_PADDED * sizeof(half)));
    
    // Convert weights
    {
        int block = 256;
        int grid1 = (FC1_OUT * FC1_IN + block - 1) / block;
        int grid2 = (FC2_OUT * FC2_IN + block - 1) / block;
        convert_weights_padded_kernel<<<grid1, block>>>(d_fc1_w, d_fc1_w_half, FC1_OUT, FC1_IN, FC1_IN_PADDED);
        convert_weights_padded_kernel<<<grid2, block>>>(d_fc2_w, d_fc2_w_half, FC2_OUT, FC2_IN, FC2_IN_PADDED);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    
    // =========================================================================
    // 2. Setup streams and memory
    // =========================================================================
    const int NUM_STREAMS = 2;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaStreamCreate(&streams[i]));
    }
    
    float* h_input[NUM_STREAMS];
    float* h_output[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaMallocHost(&h_input[i], BATCH_SIZE * IN_SIZE * sizeof(float)));
        checkCudaErrors(cudaMallocHost(&h_output[i], BATCH_SIZE * FC3_OUT * sizeof(float)));
    }
    
    size_t workspace_size = BATCH_SIZE * (IN_SIZE + C1_SIZE * 2 + P1_SIZE + C2_SIZE * 2 + P2_SIZE + 
                                          FC1_OUT * 2 + FC2_OUT * 2 + FC3_OUT);
    float* d_workspace[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        checkCudaErrors(cudaMalloc(&d_workspace[i], workspace_size * sizeof(float)));
    }
    
    // =========================================================================
    // 3. Create CUDA Graphs for the time-step loop (one per stream)
    // =========================================================================
    cudaGraph_t graphs[NUM_STREAMS];
    cudaGraphExec_t graphExecs[NUM_STREAMS];
    bool graphsCaptured[NUM_STREAMS] = {false, false};
    
    // Pre-compute kernel configurations
    dim3 block1(8, 8);
    dim3 grid1((C1_W + 7) / 8, (C1_H + 7) / 8, BATCH_SIZE);
    
    dim3 block_p1(16, 8);
    dim3 grid_p1((P1_W + 15) / 16, (P1_H + 7) / 8, BATCH_SIZE * C1_OUT);
    
    dim3 block2(8, 8);
    dim3 grid2(1, 1, BATCH_SIZE);
    
    dim3 block_p2(4, 4);
    dim3 grid_p2(1, 1, BATCH_SIZE * C2_OUT);
    
    const int FC1_WMMA_TILE = 16;
    dim3 block_fc1(32);
    dim3 grid_fc1((FC1_OUT + FC1_WMMA_TILE - 1) / FC1_WMMA_TILE, (BATCH_SIZE + FC1_WMMA_TILE - 1) / FC1_WMMA_TILE);
    
    const int FC2_WMMA_TILE = 16;
    dim3 block_fc2(32);
    dim3 grid_fc2((FC2_OUT + FC2_WMMA_TILE - 1) / FC2_WMMA_TILE, (BATCH_SIZE + FC2_WMMA_TILE - 1) / FC2_WMMA_TILE);
    
    dim3 block_fc3(32);
    dim3 grid_fc3(1, BATCH_SIZE);
    
    std::vector<int> predictions(total_images);
    
    // =========================================================================
    // 4. Main inference loop
    // =========================================================================
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int sid = batch_idx % NUM_STREAMS;
        cudaStream_t stream = streams[sid];
        
        // Pipelining: Process previous batch output
        if (batch_idx >= NUM_STREAMS) {
            int prev_idx = batch_idx - NUM_STREAMS;
            checkCudaErrors(cudaStreamSynchronize(stream));
            
            int prev_start = prev_idx * BATCH_SIZE;
            int prev_size = std::min(BATCH_SIZE, total_images - prev_start);
            
            for (int i = 0; i < prev_size; i++) {
                float* ptr = h_output[sid] + i * FC3_OUT;
                predictions[prev_start + i] = std::distance(ptr, std::max_element(ptr, ptr + FC3_OUT));
            }
        }
        
        int start = batch_idx * BATCH_SIZE;
        int current_size = std::min(BATCH_SIZE, total_images - start);
        
        // CPU Data Prep
        for (int i = 0; i < current_size; i++) {
            memcpy(h_input[sid] + i * IN_SIZE, images[start + i].data(), IN_SIZE * sizeof(float));
        }
        
        // Workspace Pointers
        float* d_ptr = d_workspace[sid];
        float* d_input = d_ptr; d_ptr += BATCH_SIZE * IN_SIZE;
        float* d_c1 = d_ptr; d_ptr += BATCH_SIZE * C1_SIZE;
        float* d_v1 = d_ptr; d_ptr += BATCH_SIZE * C1_SIZE;
        float* d_p1 = d_ptr; d_ptr += BATCH_SIZE * P1_SIZE;
        float* d_c2 = d_ptr; d_ptr += BATCH_SIZE * C2_SIZE;
        float* d_v2 = d_ptr; d_ptr += BATCH_SIZE * C2_SIZE;
        float* d_p2 = d_ptr; d_ptr += BATCH_SIZE * P2_SIZE;
        float* d_fc1 = d_ptr; d_ptr += BATCH_SIZE * FC1_OUT;
        float* d_v3 = d_ptr; d_ptr += BATCH_SIZE * FC1_OUT;
        float* d_fc2 = d_ptr; d_ptr += BATCH_SIZE * FC2_OUT;
        float* d_v4 = d_ptr; d_ptr += BATCH_SIZE * FC2_OUT;
        float* d_output = d_ptr;
        
        // Copy input
        checkCudaErrors(cudaMemcpyAsync(d_input, h_input[sid], 
            current_size * IN_SIZE * sizeof(float), cudaMemcpyHostToDevice, stream));
        
        // Init Potentials
        checkCudaErrors(cudaMemsetAsync(d_v1, 0, current_size * C1_SIZE * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_v2, 0, current_size * C2_SIZE * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_v3, 0, current_size * FC1_OUT * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_v4, 0, current_size * FC2_OUT * sizeof(float), stream));
        checkCudaErrors(cudaMemsetAsync(d_output, 0, current_size * FC3_OUT * sizeof(float), stream));
        
        // Use CUDA Graph for full batch, otherwise run kernels directly
        if (current_size == BATCH_SIZE) {
            if (!graphsCaptured[sid]) {
                // Capture graph on first full batch
                checkCudaErrors(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                
                for (int t = 0; t < T; t++) {
                    // Conv1 + IF
                    fused_conv_if_kernel1_ptx<<<grid1, block1, 0, stream>>>(
                        d_input, d_conv1_w, d_conv1_b, d_v1, d_c1, BATCH_SIZE);
                    
                    // Pool1
                    maxpool2d_kernel_ptx<<<grid_p1, block_p1, 0, stream>>>(
                        d_c1, d_p1, BATCH_SIZE, C1_OUT, C1_H, C1_W, P1_H, P1_W);
                    
                    // Conv2 + IF
                    fused_conv_if_kernel2_ptx<<<grid2, block2, 0, stream>>>(
                        d_p1, d_conv2_w, d_conv2_b, d_v2, d_c2, BATCH_SIZE);
                    
                    // Pool2
                    maxpool2d_kernel_ptx<<<grid_p2, block_p2, 0, stream>>>(
                        d_c2, d_p2, BATCH_SIZE, C2_OUT, C2_H, C2_W, P2_H, P2_W);
                    
                    // FC1 + IF (with half weights)
                    wmma_linear_if_kernel_half<FC1_IN, FC1_OUT, FC1_IN_PADDED><<<grid_fc1, block_fc1, 0, stream>>>(
                        d_p2, d_fc1_w_half, d_fc1_b, d_v3, d_fc1, BATCH_SIZE);
                    
                    // FC2 + IF (with half weights)
                    wmma_linear_if_kernel_half<FC2_IN, FC2_OUT, FC2_IN_PADDED><<<grid_fc2, block_fc2, 0, stream>>>(
                        d_fc1, d_fc2_w_half, d_fc2_b, d_v4, d_fc2, BATCH_SIZE);
                    
                    // FC3
                    linear_accumulate_fc3_kernel<<<grid_fc3, block_fc3, 0, stream>>>(
                        d_fc2, d_fc3_w, d_fc3_b, d_output, BATCH_SIZE);
                }
                
                checkCudaErrors(cudaStreamEndCapture(stream, &graphs[sid]));
                checkCudaErrors(cudaGraphInstantiate(&graphExecs[sid], graphs[sid], NULL, NULL, 0));
                graphsCaptured[sid] = true;
            }
            
            // Launch the captured graph
            checkCudaErrors(cudaGraphLaunch(graphExecs[sid], stream));
            
        } else {
            // For partial batches, run kernels directly with adjusted grid sizes
            dim3 grid1_adj((C1_W + 7) / 8, (C1_H + 7) / 8, current_size);
            dim3 grid_p1_adj((P1_W + 15) / 16, (P1_H + 7) / 8, current_size * C1_OUT);
            dim3 grid2_adj(1, 1, current_size);
            dim3 grid_p2_adj(1, 1, current_size * C2_OUT);
            dim3 grid_fc1_adj((FC1_OUT + FC1_WMMA_TILE - 1) / FC1_WMMA_TILE, (current_size + FC1_WMMA_TILE - 1) / FC1_WMMA_TILE);
            dim3 grid_fc2_adj((FC2_OUT + FC2_WMMA_TILE - 1) / FC2_WMMA_TILE, (current_size + FC2_WMMA_TILE - 1) / FC2_WMMA_TILE);
            dim3 grid_fc3_adj(1, current_size);
            
            for (int t = 0; t < T; t++) {
                fused_conv_if_kernel1_ptx<<<grid1_adj, block1, 0, stream>>>(
                    d_input, d_conv1_w, d_conv1_b, d_v1, d_c1, current_size);
                
                maxpool2d_kernel_ptx<<<grid_p1_adj, block_p1, 0, stream>>>(
                    d_c1, d_p1, current_size, C1_OUT, C1_H, C1_W, P1_H, P1_W);
                
                fused_conv_if_kernel2_ptx<<<grid2_adj, block2, 0, stream>>>(
                    d_p1, d_conv2_w, d_conv2_b, d_v2, d_c2, current_size);
                
                maxpool2d_kernel_ptx<<<grid_p2_adj, block_p2, 0, stream>>>(
                    d_c2, d_p2, current_size, C2_OUT, C2_H, C2_W, P2_H, P2_W);
                
                wmma_linear_if_kernel_half<FC1_IN, FC1_OUT, FC1_IN_PADDED><<<grid_fc1_adj, block_fc1, 0, stream>>>(
                    d_p2, d_fc1_w_half, d_fc1_b, d_v3, d_fc1, current_size);
                
                wmma_linear_if_kernel_half<FC2_IN, FC2_OUT, FC2_IN_PADDED><<<grid_fc2_adj, block_fc2, 0, stream>>>(
                    d_fc1, d_fc2_w_half, d_fc2_b, d_v4, d_fc2, current_size);
                
                linear_accumulate_fc3_kernel<<<grid_fc3_adj, block_fc3, 0, stream>>>(
                    d_fc2, d_fc3_w, d_fc3_b, d_output, current_size);
            }
        }
        
        checkCudaErrors(cudaMemcpyAsync(h_output[sid], d_output,
            current_size * FC3_OUT * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
    
    // Process remaining batches
    for (int i = 0; i < std::min(NUM_STREAMS, num_batches); i++) {
        int last_idx = num_batches - std::min(NUM_STREAMS, num_batches) + i;
        int sid = last_idx % NUM_STREAMS;
        
        checkCudaErrors(cudaStreamSynchronize(streams[sid]));
        
        int last_start = last_idx * BATCH_SIZE;
        int last_size = std::min(BATCH_SIZE, total_images - last_start);
        
        for (int j = 0; j < last_size; j++) {
            float* ptr = h_output[sid] + j * FC3_OUT;
            predictions[last_start + j] = std::distance(ptr, std::max_element(ptr, ptr + FC3_OUT));
        }
    }
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        if (graphsCaptured[i]) {
            checkCudaErrors(cudaGraphExecDestroy(graphExecs[i]));
            checkCudaErrors(cudaGraphDestroy(graphs[i]));
        }
        checkCudaErrors(cudaStreamDestroy(streams[i]));
        checkCudaErrors(cudaFreeHost(h_input[i]));
        checkCudaErrors(cudaFreeHost(h_output[i]));
        checkCudaErrors(cudaFree(d_workspace[i]));
    }
    
    checkCudaErrors(cudaFree(d_fc1_w_half));
    checkCudaErrors(cudaFree(d_fc2_w_half));
    
    return predictions;
}

// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_model_and_data_dir>" << std::endl;
        return 1;
    }
	std::string dir = argv[1];
	
    // Load test data
    auto images = read_mnist_images(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-images-idx3-ubyte");
    auto labels = read_mnist_labels(dir + "/../../.." + "/data/FashionMNIST/raw/t10k-labels-idx1-ubyte");
    if (images.empty() || labels.empty()) return 1;

    // Load model parameters to host memory
    auto conv1_w = read_param(dir + "/conv1.weight.txt");
    auto conv1_b = read_param(dir + "/conv1.bias.txt");
    auto conv2_w = read_param(dir + "/conv2.weight.txt");
    auto conv2_b = read_param(dir + "/conv2.bias.txt");
    auto fc1_w = read_param(dir + "/fc1.weight.txt");
    auto fc1_b = read_param(dir + "/fc1.bias.txt");
    auto fc2_w = read_param(dir + "/fc2.weight.txt");
    auto fc2_b = read_param(dir + "/fc2.bias.txt");
    auto fc3_w = read_param(dir + "/fc3.weight.txt");
    auto fc3_b = read_param(dir + "/fc3.bias.txt");
    
    // --- 1. Allocate all necessary GPU memory ---
    // Device pointers for parameters
    float *d_conv1_w, *d_conv1_b, *d_conv2_w, *d_conv2_b;
    float *d_fc1_w, *d_fc1_b, *d_fc2_w, *d_fc2_b, *d_fc3_w, *d_fc3_b;

    // Allocate parameters
    checkCudaErrors(cudaMalloc(&d_conv1_w, conv1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv1_b, conv1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_w, conv2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_conv2_b, conv2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_w,   fc1_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc1_b,   fc1_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_w,   fc2_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_b,   fc2_b.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_w,   fc3_w.size() * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_b,   fc3_b.size() * sizeof(float)));

    // --- 2. Copy constant parameters from host to device ---
    checkCudaErrors(cudaMemcpy(d_conv1_w, conv1_w.data(), conv1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv1_b, conv1_b.data(), conv1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_w, conv2_w.data(), conv2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_conv2_b, conv2_b.data(), conv2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_w, fc1_w.data(), fc1_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc1_b, fc1_b.data(), fc1_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_w, fc2_w.data(), fc2_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc2_b, fc2_b.data(), fc2_b.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_w, fc3_w.data(), fc3_w.size() * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_fc3_b, fc3_b.data(), fc3_b.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();
    
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================

    // --- 3. Perform inference ---
    // Pass device pointers to the inference function
    std::vector<int> predictions = scnn_inference(images,
        d_conv1_w, d_conv1_b, d_conv2_w, d_conv2_b,
        d_fc1_w, d_fc1_b, d_fc2_w, d_fc2_b, d_fc3_w, d_fc3_b
        // YOU CAN ADD MORE PARAMETERS HERE!!!
        );
    
// ===================================================================================
// Main Function -  DO NOT MODIFY BEGIN
// ===================================================================================

    // Synchronize to ensure all GPU work is done before stopping the timer
    checkCudaErrors(cudaDeviceSynchronize());
    
    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    
    // --- 4. Free all allocated GPU memory ---
    checkCudaErrors(cudaFree(d_conv1_w));
    checkCudaErrors(cudaFree(d_conv1_b));
    checkCudaErrors(cudaFree(d_conv2_w));
    checkCudaErrors(cudaFree(d_conv2_b));
    checkCudaErrors(cudaFree(d_fc1_w));
    checkCudaErrors(cudaFree(d_fc1_b));
    checkCudaErrors(cudaFree(d_fc2_w));
    checkCudaErrors(cudaFree(d_fc2_b));
    checkCudaErrors(cudaFree(d_fc3_w));
    checkCudaErrors(cudaFree(d_fc3_b));
    
    // Calculate accuracy
    int correct_predictions = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (predictions[i] == labels[i]) {
            correct_predictions++;
        }
    }
    double accuracy = static_cast<double>(correct_predictions) / labels.size();
    
    // Output result in the required format
    std::cout << std::fixed << std::setprecision(4) << diff.count() << ":" << accuracy << std::endl;
    
    return 0;
}
// ===================================================================================
// Main Function -  DO NOT MODIFY END
// ===================================================================================