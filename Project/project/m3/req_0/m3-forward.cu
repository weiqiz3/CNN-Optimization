#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "matmul.h"
#define PERMUTE_BLOCK_SIZE 256
#define TILE_WIDTH 16
#define NUM_STREAM 4
__global__ void matrix_unrolling_kernel(const float *input, float *output,
                                        const int Batch, const int Channel,
                                        const int Height, const int Width,
                                        const int K) {
    /*
    Modify this function to implement the input matrix unrolling kernel.
    Function paramter definitions:
    input - input
    output - output
    Batch - batch_size (number of images in x)
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working
    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    // TODO: Insert your input matrix unrolling kernel code here
    // const int Height_unrolled = Channel * K * K;
    // const int Width_unrolled = Batch * Height_out * Width_out;
    const int by = blockIdx.y, bx = blockIdx.x, ty = threadIdx.y, tx = threadIdx.x;
    const int ww = ceil((float) Width_out / TILE_WIDTH);
    int h = bx / ww * TILE_WIDTH + ty;
    int w = bx % ww * TILE_WIDTH + tx;
    if (h < Height_out && w < Width_out) {
        for (int c = 0; c < Channel; c++) {
            int w_base = c * K * K;
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    int h_unroll = w_base + p * K + q;
                    int w_unroll = h * Width_out + by * Height_out * Width_out + w;
                    output[((size_t) h_unroll * Batch * Height_out * Width_out + (size_t) w_unroll)] = in_4d(by, c, h + p, w + q);
                }
            }
        }
    }
    #undef in_4d
}
// Permutes the matmul result.
// The output feature map after matmul is of shape Map_out x Batch x Height_out x Width_out,
// and we need to permute it into Batch x Map_out x Height_out x Width_out.
// You don't need to modify this kernel.
__global__ void matrix_permute_kernel(const float *input, float *output, int Map_out,
                                      int Batch, int image_size) {
    int b = blockIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < image_size) {
        for (int m = 0; m < Map_out; m++) {
            output[b * Map_out * image_size + m * image_size + x] =
                    input[m * Batch * image_size + b * image_size + x];
        }
    }
}
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // TODO: Allocate memory and copy over the relevant data structures to the GPU
    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    // Useful snippet for error checking
    cudaMalloc((void **) device_input_ptr, Batch * Channel * Height * Width * sizeof(float));
    cudaMalloc((void **) device_output_ptr, Batch * Map_out * (Height - K + 1) * (Width - K + 1) * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, Map_out * Channel * K * K * sizeof(float));
    cudaMemcpy(*device_mask_ptr, host_mask, Map_out * Channel * K * K * sizeof(float), cudaMemcpyHostToDevice);
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Height_unrolled = Channel * K * K;
    const int Width_unrolled = Batch * Height_out * Width_out;
    float *unrolled_matrix;  // Pointer to device memory for storing the unrolled matrix
    float *matmul_output;    // Pointer to device memory for storing the result of matrix multiplication
    cudaMalloc((void**)&unrolled_matrix, (size_t) Height_unrolled * Width_unrolled * sizeof(float));
    cudaMalloc((void**)&matmul_output, (Batch * Map_out * Height_out * Width_out) * sizeof(float));
    cudaHostRegister((void *) host_input, Batch * Channel * Height * Width * sizeof(float), cudaHostRegisterDefault);
    cudaHostRegister((void *) host_output, Batch * Map_out * Height_out * Width_out * sizeof(float), cudaHostRegisterDefault);
    cudaStream_t streams[NUM_STREAM];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);
    for (int i = 0; i < NUM_STREAM; i++) {
        cudaMemcpyAsync(*device_input_ptr + Batch * Channel * Height * Width * i / NUM_STREAM, host_input + Batch * Channel * Height * Width * i / NUM_STREAM, Batch * Channel * Height * Width * sizeof(float) / NUM_STREAM, cudaMemcpyHostToDevice, streams[i]);
    }
    for (int i = 0; i < NUM_STREAM; i++) {
        dim3 DimGrid(ceil((float) Height_out / TILE_WIDTH) * ceil((float) Width_out / TILE_WIDTH), Batch / NUM_STREAM, 1);
        dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        matrix_unrolling_kernel<<<DimGrid, DimBlock, 0, streams[i]>>>(*device_input_ptr + Batch * Channel * Height * Width * i / NUM_STREAM, unrolled_matrix + (size_t) Height_unrolled * Width_unrolled * i / NUM_STREAM, Batch / NUM_STREAM, Channel, Height, Width, K);
    }
    for (int i = 0; i < NUM_STREAM; i++) {
        dim3 DimGrid(ceil((float) Width_unrolled / NUM_STREAM / TILE_WIDTH), ceil((float) Map_out / TILE_WIDTH), 1);
        dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        matrixMultiplyShared<<<DimGrid, DimBlock, 0, streams[i]>>>(*device_mask_ptr, unrolled_matrix + (size_t) Height_unrolled * Width_unrolled * i / NUM_STREAM, matmul_output + Batch * Map_out * Height_out * Width_out * i / NUM_STREAM, Map_out, Height_unrolled, Height_unrolled, Width_unrolled / NUM_STREAM, Map_out, Width_unrolled / NUM_STREAM);
    }
    for (int i = 0; i < NUM_STREAM; i++) {
        dim3 DimGrid(ceil((float) Height_out * Width_out / PERMUTE_BLOCK_SIZE), Batch / NUM_STREAM, 1);
        dim3 DimBlock(PERMUTE_BLOCK_SIZE, 1, 1);
        matrix_permute_kernel<<<DimGrid, DimBlock, 0, streams[i]>>>(matmul_output + Batch * Map_out * Height_out * Width_out * i / NUM_STREAM, *device_output_ptr + Batch * Map_out * Height_out * Width_out * i / NUM_STREAM, Map_out, Batch / NUM_STREAM, Height_out * Width_out);
    }
    for (int i = 0; i < NUM_STREAM; i++) {
        cudaMemcpyAsync((void*) (host_output + Width_unrolled * Map_out * i / NUM_STREAM), *device_output_ptr + Width_unrolled * Map_out * i / NUM_STREAM, Width_unrolled * Map_out * sizeof(float) / NUM_STREAM, cudaMemcpyDeviceToHost, streams[i]);
    }
    for (int i = 0; i < NUM_STREAM; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(*device_input_ptr);
    cudaFree(*device_output_ptr);
    cudaFree(*device_mask_ptr);
    cudaFree(unrolled_matrix);
    cudaFree(matmul_output);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }
}
__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
}
__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
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