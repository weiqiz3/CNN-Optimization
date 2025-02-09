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
#define TILE_WIDTH 16
// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBRows, int numBColumns,
                                     int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  __shared__ float subTileM[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileN[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;
  float p = 0;
  for (int q = 0; q < (numAColumns - 1) / TILE_WIDTH + 1; q++) {
    if (Row < numARows && q * TILE_WIDTH + tx < numAColumns) {
      subTileM[ty][tx] = A[Row * numAColumns + q * TILE_WIDTH + tx];
    } else {
      subTileM[ty][tx] = 0;
    }
    if (Col < numBColumns && q * TILE_WIDTH + ty < numBRows) {
      subTileN[ty][tx] = B[(q * TILE_WIDTH + ty) * numBColumns + Col];
    } else {
      subTileN[ty][tx] = 0;
    }
    __syncthreads();
    if (Row < numCRows && Col < numCColumns) {
      for (int k = 0; k < TILE_WIDTH; k++) {
        p += subTileM[ty][k] * subTileN[k][tx];
      }
    }
    __syncthreads();
  }
  if (Row < numCRows && Col < numCColumns) {
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
  matrixMultiplyShared<<<DimGrid, DimBlock>>>(devA, devB, devC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
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
  //@@ Free the hostC matrix
  free(hostC);
  return 0;
}