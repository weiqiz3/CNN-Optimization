// Histogram Equalization
#include <wb.h>
#define HISTOGRAM_LENGTH 256
//@@ insert code here
__global__ void flToUnChar(float *inputImage, unsigned char *ucharImage, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    ucharImage[i] = (unsigned char) (255 * inputImage[i]);
  }
}
__global__ void rgbToGS(unsigned char *ucharImage, unsigned char *grayImage, int height, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < height * width) {
    int r = ucharImage[i * 3];
    int g = ucharImage[i * 3 + 1];
    int b = ucharImage[i * 3 + 2];
    grayImage[i] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
  }
}
__global__ void histogram(unsigned char *grayImage, unsigned int *histo, int height, int width) {
  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    histo_private[threadIdx.x] = 0;
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  while (i < height * width) {
    atomicAdd(&histo_private[grayImage[i]], 1);
    i += stride;
  }
  __syncthreads();
  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
  }
}
__global__ void CDF(unsigned int *histo, float *cdf, int height, int width) {
  __shared__ float T[2 * HISTOGRAM_LENGTH];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < HISTOGRAM_LENGTH) {
    T[threadIdx.x] = histo[i];
  }
  int stride = 1;
  while (stride <= HISTOGRAM_LENGTH) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index < HISTOGRAM_LENGTH && index >= stride) {
      T[index] += T[index - stride];
    }
    stride *= 2;
  }
  stride = HISTOGRAM_LENGTH / 2;
  while (stride > 0) {
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < HISTOGRAM_LENGTH) {
      T[index + stride] += T[index];
    }
    stride /= 2;
  }
  __syncthreads();
  if (i < HISTOGRAM_LENGTH) {
    cdf[i] = T[threadIdx.x] / height / width;
  }
}
__global__ void equalize(unsigned char *ucharImage, unsigned char *Image, float *cdf, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    // Image[i] = min(max(255.0 * (cdf[ucharImage[i]] - cdf[0]) / (1 - cdf[0]) / (HISTOGRAM_LENGTH - 1), 0.0), 255.0);
    float v = 255.0f * (cdf[ucharImage[i]] - cdf[0]) / (1.0f - cdf[0]);
    v = fminf(fmaxf(v, 0.0f), 255.0f);
    Image[i] = static_cast<unsigned char>(v + 0.5f);
  }
}
__global__ void ucToFloat(unsigned char *ucharImage, float *outputImage, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    outputImage[i] = (float) ucharImage[i] / 255;
  }
}
int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;
  //@@ Insert more code here
  float *devIn, *devOut, *cdf;
  unsigned char *devUC, *grayImage, *eqimage;
  unsigned int *hist;
  args = wbArg_read(argc, argv); /* parse the input arguments */
  inputImageFile = wbArg_getInputFile(args, 0);
  //Import data and create memory on host
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  //@@ insert code here
  cudaMalloc((void **) &devIn, imageHeight * imageWidth * imageChannels * sizeof(float));
  cudaMalloc((void **) &devOut, imageHeight * imageWidth * imageChannels * sizeof(float));
  cudaMalloc((void **) &cdf, HISTOGRAM_LENGTH * sizeof(float));
  cudaMalloc((void **) &devUC, imageHeight * imageWidth * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &grayImage, imageHeight * imageWidth * sizeof(unsigned char));
  cudaMalloc((void **) &eqimage, imageHeight * imageWidth * imageChannels * sizeof(unsigned char));
  cudaMalloc((void **) &hist, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemcpy(devIn, hostInputImageData, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(hist, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset(cdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  dim3 DimGrid(ceil((float) imageHeight * imageWidth * imageChannels / HISTOGRAM_LENGTH), 1, 1);
  dim3 DimBlock(HISTOGRAM_LENGTH, 1, 1);
  flToUnChar<<<DimGrid, DimBlock>>>(devIn, devUC, imageHeight * imageWidth * imageChannels);
  rgbToGS<<<DimGrid, DimBlock>>>(devUC, grayImage, imageHeight, imageWidth);
  histogram<<<DimGrid, DimBlock>>>(grayImage, hist, imageHeight, imageWidth);
  CDF<<<DimGrid, DimBlock>>>(hist, cdf, imageHeight, imageWidth);
  equalize<<<DimGrid, DimBlock>>>(devUC, eqimage, cdf, imageHeight * imageWidth * imageChannels);
  ucToFloat<<<DimGrid, DimBlock>>>(eqimage, devOut, imageHeight * imageWidth * imageChannels);
  cudaMemcpy(hostOutputImageData, devOut, imageHeight * imageWidth * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);
  //@@ insert code here
  cudaFree(devIn);
  cudaFree(devOut);
  cudaFree(cdf);
  cudaFree(devUC);
  cudaFree(grayImage);
  cudaFree(hist);
  free(hostInputImageData);
  free(hostOutputImageData);
  // free(inputImageFile);
  return 0;
}