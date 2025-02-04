#include <wb.h>
#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)
// Compute C = A * B
__global__ void matrixMultiply(float *A, float *B, float *C, int numARows,
                               int numAColumns, int numBRows,
                               int numBColumns, int numCRows,
                               int numCColumns)
{
  //@@ Implement matrix multiplication kernel here
  // Calculate the row index of the d_P element and d_M
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  // Calculate the column idenx of d_P and d_N
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((Row < numCRows) && (Col < numCColumns)) {
    float p = 0;
    // each thread computes one element of the block sub-matrix
    for (int k = 0; k < numAColumns; k++) {
      p += A[Row * numAColumns + k] * B[k * numBColumns + Col];
    }
    C[Row * numCColumns + Col] = p;
  }
}
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  args = wbArg_read(argc, argv);
  //@@ Importing data and creating memory on host
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  hostC = (float *) malloc(numCColumns * numCRows * sizeof(float));
  wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns);
  //@@ Allocate GPU memory here
  float *devA, *devB, *devC;
  cudaMalloc((void **) &devA, numAColumns * numARows * sizeof(float));
  cudaMalloc((void **) &devB, numBColumns * numBRows * sizeof(float));
  cudaMalloc((void **) &devC, numCColumns * numCRows * sizeof(float));
  //@@ Copy memory to the GPU here
  cudaMemcpy(devA, hostA, numAColumns * numARows * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(devB, hostB, numBColumns * numBRows * sizeof(float), cudaMemcpyHostToDevice);
  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(ceil((float) numCColumns / 16), ceil((float) numCRows / 16), 1);
  dim3 DimBlock(16, 16, 1);
  //@@ Launch the GPU Kernel here
  matrixMultiply<<<DimGrid, DimBlock>>>(devA, devB, devC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, devC, numCColumns * numCRows * sizeof(float), cudaMemcpyDeviceToHost);
  //@@ Free the GPU memory here
  cudaFree(devA);
  cudaFree(devB);
  cudaFree(devC);
  wbSolution(args, hostC, numCRows, numCColumns);
  free(hostA);
  free(hostB);
  //@@Free the hostC matrix
  free(hostC);
  return 0;
}