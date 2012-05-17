#include <unistd.h>
#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

using namespace std;

int main(int argc, char ** argv){

  int status;
  int lower = 2;
  int upper = 100;
  int num = 100000;
  int reps = 5;
  
  while((status = getopt(argc, argv, "l:u:n:r:")) != -1){
    switch(status){
    case 'l':
      lower = strtoul(optarg, 0, 0);
      break;
    case 'u':
      upper = strtoul(optarg, 0, 0);
      break;
    case 'n':
      num = strtoul(optarg, 0, 0);
      break;
    case 'r':
      reps = strtoul(optarg, 0, 0);
      break;
    default:
      cerr << "invalid argument: " << status << endl;
      exit(1);
    }
  }

  float *matrices = (float*)malloc(upper * upper * num * sizeof(float));
  float *vectors = (float*)malloc(upper * num * sizeof(float));

  for(int i = 0; i < num * upper * upper; i++)
    matrices[i] = drand48();

  for(int i = 0; i < num * upper; i++)
    vectors[i] = drand48();

  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;

  stat = cublasCreate(&handle);
  if(stat != CUBLAS_STATUS_SUCCESS){
    cerr << "cublas init failed" << endl;
    exit(1);
  }

  // allocate input space on device
  struct cudaPitchedPtr devMatrices;
  struct cudaExtent devMatricesExtent;
  
  devMatricesExtent = 
    make_cudaExtent(upper * sizeof(float),
		    upper,
		    num);

  cudaStat = 
    cudaMalloc3D(&devMatrices,
		 devMatricesExtent);

  float *devVectors = 0;
  size_t devVectorsPitch;
  cudaStat = 
    cudaMallocPitch(
		    &devVectors,
		    &devVectorsPitch,
		    upper * sizeof(float),
		    num);

  // allocate result space on device
  float *devResult = 0;
  size_t devResultPitch;
  cudaStat = 
    cudaMallocPitch(
		    &devResult,
		    &devResultPitch,
		    upper * sizeof(float),
		    num
		    );

  // copy data to device
  struct cudaMemcpy3DParms devMatricesParams = {0};
  devMatricesParams.extent = devMatricesExtent;
  devMatricesParams.kind = cudaMemcpyHostToDevice;
  devMatricesParams.dstPtr = devMatrices;
  devMatricesParams.srcPtr = 
    make_cudaPitchedPtr(matrices, 
		      upper * sizeof(float),
		      upper,
		      num);

  cudaStat = 
    cudaMemcpy3D(&devMatricesParams);
  
  cudaStat = 
    cudaMemcpy2D(devVectors,
		 devVectorsPitch,
		 vectors,
		 upper * sizeof(float),
		 upper * sizeof(float),
		 num,
		 cudaMemcpyHostToDevice);

  // create lists of device pointers to inputs and outputs
  float **AList = 0, **BList = 0, **CList = 0;

  AList = (float**)malloc(num * sizeof(float*));
  BList = (float**)malloc(num * sizeof(float*));
  CList = (float**)malloc(num * sizeof(float*));

  for(int i = 0; i < num; i++){
    AList[i] = (float*)devMatrices.ptr + 
      devMatrices.pitch/sizeof(float) * devMatrices.ysize * i;
    BList[i] = devVectors + devVectorsPitch/sizeof(float) * i;
    CList[i] = devResult + devResultPitch/sizeof(float) * i;
  }

  // copy pointer lists to device
  float **devAList = 0, **devBList = 0, **devCList = 0;
  cudaStat = cudaMalloc(&devAList, num * sizeof(float*));
  cudaStat = cudaMalloc(&devBList, num * sizeof(float*));
  cudaStat = cudaMalloc(&devCList, num * sizeof(float*));

  cudaStat = cudaMemcpy(devAList,
			AList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);
  
  cudaStat = cudaMemcpy(devBList,
			BList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);

  cudaStat = cudaMemcpy(devCList,
			CList,
			num * sizeof(float*),
			cudaMemcpyHostToDevice);

  // perform <num> <size x size> x <size x 1> multiplications
  for(int size = lower; size <= upper; size++){
    
    /*
    stat = cublasSgemmBatched(handle,
			      CUBLAS_OP_N,
			      CUBLAS_OP_N,
			      
			      );
    */
  }
  free(matrices);
  free(vectors);
      
  return 0;
}