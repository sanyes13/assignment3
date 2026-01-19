#include <iostream>
#include <cuda_runtime.h>

// Задание 1: Умножение на число (Глобальная vs Разделяемая память)
__global__ void multiply_global(float* data, float factor, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= factor;
}

__global__ void multiply_shared(float* data, float factor, int n) {
    extern __shared__ float s_data[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < n) s_data[tid] = data[idx];
    __syncthreads();

    if (idx < n) {
        s_data[tid] *= factor;
        data[idx] = s_data[tid];
    }
}

// Задание 2: Поэлементное сложение
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// Задание 3: Коалесцированный vs Некоалесцированный доступ
__global__ void coalesced_access(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] += 1.0f;
}

__global__ void non_coalesced_access(float* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride % n;
    data[idx] += 1.0f;
}

void run_benchmarks() {
    int n = 1000000; // [cite: 68, 74]
    size_t size = n * sizeof(float);
    float *h_data = new float[n], *d_data;
    cudaMalloc(&d_data, size);

    // Тест Задания 1
    multiply_global<<<(n+255)/256, 256>>>(d_data, 2.0f, n);
    multiply_shared<<<(n+255)/256, 256, 256*sizeof(float)>>>(d_data, 2.0f, n);

    // Тест Задания 2 (Разные размеры блоков) [cite: 71]
    int blocks[] = {128, 256, 512};
    for(int b : blocks) {
        vector_add<<<(n+b-1)/b, b>>>(d_data, d_data, d_data, n);
        std::cout << "Block size " << b << " executed." << std::endl;
    }

    // Тест Задания 3
    coalesced_access<<<(n+255)/256, 256>>>(d_data, n);
    non_coalesced_access<<<(n+255)/256, 256>>>(d_data, n, 32);

    cudaFree(d_data);
    delete[] h_data;
}

int main() {
    run_benchmarks();
    return 0;
}
