#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <algorithm>

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
            images[i][j] = (static_cast<float>(buffer[j]) / 255.0f - 0.5f) / 0.5f; // Normalization
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
// CUDA KERNELS BEGIN
// ===================================================================================

// 2D 卷积核函数
__global__ void conv2d_kernel(
    const float* input, 
    float* output, 
    const float* weights, 
    const float* biases,                          
    int in_channels, 
    int out_channels, 
    int in_height, 
    int in_width,                            
    int kernel_size) {

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = blockIdx.z;

    int out_width = in_width - kernel_size + 1;
    int out_height = in_height - kernel_size + 1;

    // out_x in [0, 24) out_y in [0, 24) out_c in [0, 6)
    if (out_x < out_width && out_y < out_height && out_c < out_channels) {
        float sum = biases[out_c];
        for (int in_c = 0; in_c < in_channels; ++in_c) {
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    
                    int weight_idx = out_c * (in_channels * kernel_size * kernel_size) +
                                     in_c * (kernel_size * kernel_size) +
                                     ky * kernel_size + 
                                     kx;
                    int input_idx = in_c * (in_height * in_width) +
                                    in_y * in_width + 
                                    in_x;

                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
        int output_idx = out_c * (out_height * out_width) + 
                         out_y * out_width + 
                         out_x;
        output[output_idx] = sum;
    }
}


// 最大池化核函数 (2x2)
__global__ void max_pool_2x2_kernel(
    const float* input, 
    float* output,
    int channels, 
    int in_height, 
    int in_width) {

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.z;

    int out_width = in_width / 2;
    int out_height = in_height / 2;

    if (out_x < out_width && out_y < out_height && c < channels) {
        int in_start_y = out_y * 2;
        int in_start_x = out_x * 2;

        float max_val = -1e20f;
        for (int ky = 0; ky < 2; ++ky) {
            for (int kx = 0; kx < 2; ++kx) {
                int in_y = in_start_y + ky;
                int in_x = in_start_x + kx;
                int input_idx = c * (in_height * in_width) + 
                                in_y * in_width + 
                                in_x;
                max_val = fmaxf(max_val, input[input_idx]);
            }
        }
        int output_idx = c * (out_height * out_width) + 
                         out_y * out_width + 
                         out_x;
        output[output_idx] = max_val;
    }
}

// 全连接 (线性) 层核函数
__global__ void linear_kernel(
    const float* input, 
    float* output, 
    const float* weights, 
    const float* biases,                  
    int in_features, 
    int out_features) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out_features) {
        float sum = biases[i];
        for (int j = 0; j < in_features; ++j) {
            sum += input[j] * weights[i * in_features + j];
        }
        output[i] = sum;
    }
}

// 积分-发放 (IF) 神经元核函数
__global__ void if_neuron_kernel(
    const float* input, 
    float* output, 
    float* potential, 
    int size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        potential[i] += input[i];
        if (potential[i] >= 1.0f) {
            output[i] = 1.0f;
            potential[i] = 0.0f; // 重置电位
        } else {
            output[i] = 0.0f;
        }
    }
}

// 为最终层累积电位（无重置）的核函数
__global__ void accumulate_potential_kernel(
    const float* input, 
    float* potential, 
    int size) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        potential[i] += input[i];
    }
}

// ===================================================================================
// CUDA KERNELS END
// ===================================================================================

// ===================================================================================
// INFERENCE FUNCTION BEGIN
// ===================================================================================
std::vector<int> scnn_inference(
    const std::vector<std::vector<float>>& images,
    // Device pointers for parameters
    float* d_conv1_w, float* d_conv1_b, float* d_conv2_w, float* d_conv2_b,
    float* d_fc1_w,   float* d_fc1_b,   float* d_fc2_w,   float* d_fc2_b,
    float* d_fc3_w,   float* d_fc3_b
    // YOU CAN ADD MORE PARAMETERS HERE!!!
    )
{
    std::vector<int> predictions;
    const int num_images = images.size();
    predictions.reserve(num_images);

    // SNN-specific parameter, must match training
    const int T = 8;
    

    // --- 定义网络各层维度 ---
    const int IMG_SIZE = 28 * 28;
    const int C1_OUT_CH = 6;  const int C1_OUT_DIM = 24; const int C1_OUT_SIZE = C1_OUT_CH * C1_OUT_DIM * C1_OUT_DIM;
    const int P1_OUT_DIM = 12; const int P1_OUT_SIZE = C1_OUT_CH * P1_OUT_DIM * P1_OUT_DIM;
    const int C2_OUT_CH = 16; const int C2_OUT_DIM = 8;  const int C2_OUT_SIZE = C2_OUT_CH * C2_OUT_DIM * C2_OUT_DIM;
    const int P2_OUT_DIM = 4;  const int P2_OUT_SIZE = C2_OUT_CH * P2_OUT_DIM * P2_OUT_DIM;
    const int FC1_IN_FEAT = P2_OUT_SIZE; const int FC1_OUT_FEAT = 120;
    const int FC2_OUT_FEAT = 84;
    const int FC3_OUT_FEAT = 10;

    // --- 分配用于存储激活值和电位的 GPU 内存 ---
    float *d_input, *d_c1, *d_p1, *d_c2, *d_p2, *d_fc1_in, *d_fc1_out, *d_fc2_out, *d_fc3_out;
    float *d_pot_if1, *d_pot_if2, *d_pot_if3, *d_pot_if4, *d_pot_fc3;

    checkCudaErrors(cudaMalloc(&d_input,    IMG_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c1,       C1_OUT_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_p1,       P1_OUT_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c2,       C2_OUT_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_p2,       P2_OUT_SIZE * sizeof(float)));
    d_fc1_in = d_p2; // 展平操作只是改变视图，可复用指针
    checkCudaErrors(cudaMalloc(&d_fc1_out,  FC1_OUT_FEAT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc2_out,  FC2_OUT_FEAT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_fc3_out,  FC3_OUT_FEAT * sizeof(float)));

    // IF 神经元的电位
    checkCudaErrors(cudaMalloc(&d_pot_if1, C1_OUT_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pot_if2, C2_OUT_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pot_if3, FC1_OUT_FEAT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pot_if4, FC2_OUT_FEAT * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_pot_fc3, FC3_OUT_FEAT * sizeof(float)));

    // 用于存储最终输出的主机端缓冲区
    std::vector<float> h_output(FC3_OUT_FEAT);


    // --- Loop over each image ---
    for (int i = 0; i < num_images; ++i) {
        
        // 将图片从主机复制到设备
        checkCudaErrors(cudaMemcpy(d_input, images[i].data(), IMG_SIZE * sizeof(float), cudaMemcpyHostToDevice));

        // 为新图片将所有电位重置为零
        checkCudaErrors(cudaMemset(d_pot_if1, 0, C1_OUT_SIZE * sizeof(float)));
        checkCudaErrors(cudaMemset(d_pot_if2, 0, C2_OUT_SIZE * sizeof(float)));
        checkCudaErrors(cudaMemset(d_pot_if3, 0, FC1_OUT_FEAT * sizeof(float)));
        checkCudaErrors(cudaMemset(d_pot_if4, 0, FC2_OUT_FEAT * sizeof(float)));
        checkCudaErrors(cudaMemset(d_pot_fc3, 0, FC3_OUT_FEAT * sizeof(float)));

        // --- 循环 T 个时间步 ---
        for (int t = 0; t < T; ++t) {
            // 第 1 层: Conv2D
            dim3 threadsPerBlockConv1(16, 16);
            dim3 numBlocksConv1((C1_OUT_DIM + 15) / 16, (C1_OUT_DIM + 15) / 16, C1_OUT_CH);
            conv2d_kernel<<<numBlocksConv1, threadsPerBlockConv1>>>(d_input, d_c1, d_conv1_w, d_conv1_b, 1, C1_OUT_CH, 28, 28, 5);

            // 第 2 层: IF Neuron 1 点对点访问，不改变特征图形状
            if_neuron_kernel<<<(C1_OUT_SIZE + 255) / 256, 256>>>(d_c1, d_c1, d_pot_if1, C1_OUT_SIZE);
            
            // 第 3 层: Max Pool 1
            dim3 threadsPerBlockPool1(16, 16);
            dim3 numBlocksPool1((P1_OUT_DIM + 15) / 16, (P1_OUT_DIM + 15) / 16, C1_OUT_CH);
            max_pool_2x2_kernel<<<numBlocksPool1, threadsPerBlockPool1>>>(d_c1, d_p1, C1_OUT_CH, C1_OUT_DIM, C1_OUT_DIM);
            
            // 第 4 层: Conv2D 2
            dim3 threadsPerBlockConv2(8, 8);
            dim3 numBlocksConv2((C2_OUT_DIM + 7) / 8, (C2_OUT_DIM + 7) / 8, C2_OUT_CH);
            conv2d_kernel<<<numBlocksConv2, threadsPerBlockConv2>>>(d_p1, d_c2, d_conv2_w, d_conv2_b, C1_OUT_CH, C2_OUT_CH, P1_OUT_DIM, P1_OUT_DIM, 5);

            // 第 5 层: IF Neuron 2
            if_neuron_kernel<<<(C2_OUT_SIZE + 255) / 256, 256>>>(d_c2, d_c2, d_pot_if2, C2_OUT_SIZE);

            // 第 6 层: Max Pool 2
            dim3 threadsPerBlockPool2(4, 4);
            dim3 numBlocksPool2((P2_OUT_DIM + 3) / 4, (P2_OUT_DIM + 3) / 4, C2_OUT_CH);
            max_pool_2x2_kernel<<<numBlocksPool2, threadsPerBlockPool2>>>(d_c2, d_p2, C2_OUT_CH, C2_OUT_DIM, C2_OUT_DIM);
            
            // 第 7 层: Flatten (展平, 通过将 d_p2 用作 d_fc1_in 隐式完成)

            // 第 8 层: Fully Connected 1
            linear_kernel<<<(FC1_OUT_FEAT + 255) / 256, 256>>>(d_fc1_in, d_fc1_out, d_fc1_w, d_fc1_b, FC1_IN_FEAT, FC1_OUT_FEAT);

            // 第 9 层: IF Neuron 3
            if_neuron_kernel<<<(FC1_OUT_FEAT + 255) / 256, 256>>>(d_fc1_out, d_fc1_out, d_pot_if3, FC1_OUT_FEAT);
            
            // 第 10 层: Fully Connected 2
            linear_kernel<<<(FC2_OUT_FEAT + 255) / 256, 256>>>(d_fc1_out, d_fc2_out, d_fc2_w, d_fc2_b, FC1_OUT_FEAT, FC2_OUT_FEAT);
            
            // 第 11 层: IF Neuron 4
            if_neuron_kernel<<<(FC2_OUT_FEAT + 255) / 256, 256>>>(d_fc2_out, d_fc2_out, d_pot_if4, FC2_OUT_FEAT);

            // 第 12 层: Fully Connected 3 (输出层)
            linear_kernel<<<(FC3_OUT_FEAT + 255) / 256, 256>>>(d_fc2_out, d_fc3_out, d_fc3_w, d_fc3_b, FC2_OUT_FEAT, FC3_OUT_FEAT);
            
            // 累积最终层的电位
            accumulate_potential_kernel<<<(FC3_OUT_FEAT + 255) / 256, 256>>>(d_fc3_out, d_pot_fc3, FC3_OUT_FEAT);
        }

        // 将最终累积的电位复制回主机
        checkCudaErrors(cudaMemcpy(h_output.data(), d_pot_fc3, FC3_OUT_FEAT * sizeof(float), cudaMemcpyDeviceToHost));
        
        // 找到最大元素的索引
        int prediction = std::distance(h_output.begin(), std::max_element(h_output.begin(), h_output.end()));





        predictions.push_back(prediction);
    }
    

    // 释放所有分配的内存
    checkCudaErrors(cudaFree(d_input));
    checkCudaErrors(cudaFree(d_c1));
    checkCudaErrors(cudaFree(d_p1));
    checkCudaErrors(cudaFree(d_c2));
    checkCudaErrors(cudaFree(d_p2));
    checkCudaErrors(cudaFree(d_fc1_out));
    checkCudaErrors(cudaFree(d_fc2_out));
    checkCudaErrors(cudaFree(d_fc3_out));
    checkCudaErrors(cudaFree(d_pot_if1));
    checkCudaErrors(cudaFree(d_pot_if2));
    checkCudaErrors(cudaFree(d_pot_if3));
    checkCudaErrors(cudaFree(d_pot_if4));
    checkCudaErrors(cudaFree(d_pot_fc3));

    // Memory is freed in main.
    
    return predictions;
}

// ===================================================================================
// INFERENCE FUNCTION END
// ===================================================================================

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
