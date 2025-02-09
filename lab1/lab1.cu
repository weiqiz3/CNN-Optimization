// LAB 1
#include <wb.h>
__global__ void vecAdd(float *in1, float *in2, float *out, int len) {
  //@@ Insert code to implement vector addition here
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    out[i] = in1[i] + in2[i];
  }
}
int main(int argc, char **argv) {
  wbArg_t args;
  int inputLength;
  float *hostInput1;
  float *hostInput2;
  float *hostOutput;
  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostInput1 =
      (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostInput2 =
      (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));
  wbLog(TRACE, "The input length is ", inputLength);
  //@@ Allocate GPU memory here
  float *devIn1, *devIn2, *devOut;
  cudaMalloc((void **) &devIn1, inputLength * sizeof(float));
  cudaMalloc((void **) &devIn2, inputLength * sizeof(float));
  cudaMalloc((void **) &devOut, inputLength * sizeof(float));
  //@@ Copy memory to the GPU here
  cudaMemcpy(devIn1, hostInput1, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devIn2, hostInput2, inputLength * sizeof(float), cudaMemcpyHostToDevice);
  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil((float) inputLength / 256), 1, 1);
  dim3 DimBlock(256, 1, 1);
  //@@ Launch the GPU Kernel here to perform CUDA computation
  vecAdd<<<DimGrid, DimBlock>>>(devIn1, devIn2, devOut, inputLength);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostOutput, devOut, inputLength * sizeof(float), cudaMemcpyDeviceToHost);
  //@@ Free the GPU memory here
  cudaFree(devIn1);
  cudaFree(devIn2);
  cudaFree(devOut);
  wbSolution(args, hostOutput, inputLength);
  free(hostInput1);
  free(hostInput2);
  free(hostOutput);
// 1969
  return 0;
}
// lol