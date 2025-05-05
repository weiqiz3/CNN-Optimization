#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <mma.h>
using namespace nvcuda;
#define TILE_WIDTH 16
#define NUM_UNROLL 4
__global__ void matmul_conv_fused(const float * __restrict__ mask, const float * __restrict__ input, float * __restrict__ output,
                                  int Batch, int Map_out, int Channel, int Height, int Width, int K)
{
    /*
    TODO: Modify this function to implement the fused unroll-matmul-permute kernel.
    Function parameter definitions:
    mask - convolution kernel
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Width_out = (Width - K) + 1;
    const int Height_out = Height - K + 1;
    int b = blockIdx.z;
    int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    int row = by * TILE_WIDTH + ty, col = bx * TILE_WIDTH + tx;
    int unrolled = Channel * K * K;
    int hw = Height_out * Width_out;
    __shared__ half tileMask[TILE_WIDTH][TILE_WIDTH];
    __shared__ half tileInput[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileOutput[TILE_WIDTH][TILE_WIDTH];
    wmma::fragment<wmma::matrix_a, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    int siu = ceil((float) unrolled / TILE_WIDTH);
    int ur = siu / NUM_UNROLL;
    for (int i = 0; i < ur; i++) {
        #pragma unroll
        for (int j = 0; j < NUM_UNROLL; j++) {
            int k = i * NUM_UNROLL + j;
            int ce = k * TILE_WIDTH + tx;
            if (row < Map_out && ce < unrolled) {
                tileMask[ty][tx] = __float2half(mask[row * unrolled + ce]);
            } else {
                tileMask[ty][tx] = __float2half(0);
            }
            int uv = k * TILE_WIDTH + ty;
            if (uv < unrolled && col < hw) {
                int c = uv / (K * K);
                int rem = uv % (K * K);
                int p = rem / K;
                int q = rem % K;
                int h = col / Width_out + p;
                int w = col % Width_out + q;
                tileInput[ty][tx] = __float2half(in_4d(b, c, h, w));
            } else {
                tileInput[ty][tx] = __float2half(0);
            }
            __syncthreads();
            if (ty < 2) {
                wmma::load_matrix_sync(a_frag, (half*) tileMask, TILE_WIDTH);
                wmma::load_matrix_sync(b_frag, (half*) tileInput, TILE_WIDTH);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }
    }
    for (int i = NUM_UNROLL * ur; i < siu; i++) {
        int uv = i * TILE_WIDTH + ty;
        if (uv < unrolled && col < hw) {
            int c = uv / (K * K);
            int rem = uv % (K * K);
            int p = rem / K;
            int q = rem % K;
            int h = col / Width_out + p;
            int w = col % Width_out + q;
            tileInput[ty][tx] = __float2half(in_4d(b, c, h, w));
        } else {
            tileInput[ty][tx] = __float2half(0);
        }
        int j = i * TILE_WIDTH + tx;
        if (row < Map_out && j < unrolled) {
            tileMask[ty][tx] = __float2half(mask[row * unrolled + j]);
        } else {
            tileMask[ty][tx] = __float2half(0);
        }
        __syncthreads();
        if (ty < 2) {
            wmma::load_matrix_sync(a_frag, (half*) tileMask, TILE_WIDTH);
            wmma::load_matrix_sync(b_frag, (half*) tileInput, TILE_WIDTH);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
        __syncthreads();
    }
    if (ty < 2) {
        wmma::store_matrix_sync((float*) tileOutput, c_frag, TILE_WIDTH, wmma::mem_row_major);
    }
    __syncthreads();
    if (row < Map_out && col < hw) {
        output[b * Map_out * hw + row * hw + col] = tileOutput[ty][tx];
    }
    #undef in_4d
}
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch * Channel * Height * Width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}
__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Set the kernel dimensions and call the fused kernel
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    dim3 DimGrid(ceil((float) Height_out * Width_out / TILE_WIDTH), ceil((float) Map_out / TILE_WIDTH), Batch);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    matmul_conv_fused<<<DimGrid, DimBlock>>>(device_mask, device_input, device_output, Batch, Map_out, Channel, Height, Width, K);
}
__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float), cudaMemcpyDeviceToHost);
    // TODO: Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
}
__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}