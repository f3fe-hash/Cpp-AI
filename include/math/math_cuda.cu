#pragma once

#include <cmath>
#include <vector>

#include <cuda_runtime.h>

#include "utils/defs.hpp"

// CUDA kernel for mult_add partial sums (dot product + add scalar)
__global__ void mult_add_kernel(const num* a, const num* b, num c, std::size_t n, num* result)
{
    extern __shared__ num sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    num tmp = 0;

    if (i < n)
        tmp = a[i] * b[i];

    sdata[tid] = tmp;
    __syncthreads();

    // Parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(result, sdata[0]);
}

inline num mult_add(const num_arr& a, const num_arr& b, const num c, std::size_t n) noexcept
{
    // Allocate device memory
    num *d_a, *d_b, *d_result;
    cudaMalloc(&d_a, n * sizeof(num));
    cudaMalloc(&d_b, n * sizeof(num));
    cudaMalloc(&d_result, sizeof(num));

    num zero_val = 0;
    cudaMemcpy(d_a, a.data(), n * sizeof(num), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(num), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &zero_val, sizeof(num), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(num);
    mult_add_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_a, d_b, c, n, d_result);

    // Copy result back
    num gpu_result = 0;
    cudaMemcpy(&gpu_result, d_result, sizeof(num), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return gpu_result + c;
}

// CUDA kernel for activation (sigmoid)
__global__ void activation_kernel(const num* input, num* output, std::size_t n)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = 1 / (1 + expf(-input[i]));
}

inline num_arr activation(const num_arr a, std::size_t n) noexcept
{
    num_arr result(n);
    num *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(num));
    cudaMalloc(&d_output, n * sizeof(num));

    cudaMemcpy(d_input, a.data(), n * sizeof(num), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    activation_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    cudaMemcpy(result.data(), d_output, n * sizeof(num), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}

// CUDA kernel for activation derivative
__global__ void activation_derv_kernel(const num* input, num* output, std::size_t n)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = input[i] * (1 - input[i]);
}

inline num_arr activation_derv(const num_arr a, std::size_t n) noexcept
{
    num_arr result(n);
    num *d_input, *d_output;

    cudaMalloc(&d_input, n * sizeof(num));
    cudaMalloc(&d_output, n * sizeof(num));

    cudaMemcpy(d_input, a.data(), n * sizeof(num), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    activation_derv_kernel<<<gridSize, blockSize>>>(d_input, d_output, n);

    cudaMemcpy(result.data(), d_output, n * sizeof(num), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}

// CUDA kernel for error calculation
__global__ void error_kernel(const num* x, const num* y, num* output, std::size_t n)
{
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        output[i] = x[i] - y[i];
}

inline num_arr error(const num_arr x, const num_arr y, std::size_t n) noexcept
{
    num_arr result(n);
    num *d_x, *d_y, *d_output;

    cudaMalloc(&d_x, n * sizeof(num));
    cudaMalloc(&d_y, n * sizeof(num));
    cudaMalloc(&d_output, n * sizeof(num));

    cudaMemcpy(d_x, x.data(), n * sizeof(num), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), n * sizeof(num), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    error_kernel<<<gridSize, blockSize>>>(d_x, d_y, d_output, n);

    cudaMemcpy(result.data(), d_output, n * sizeof(num), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_output);

    return result;
}