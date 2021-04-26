#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <cuda.h>
#include <random>

#include "csvHandler.h"
#include "definitions.h"

#include <unistd.h>
#include <chrono>
using namespace std::chrono;

using namespace std;

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

//---------------------------------------------------------------------------------------------------------------------------------------

__global__ void warm_up_gpu(){  // this kernel avoids cold start when evaluating duration of kmeans exec.
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid;
}

//----------------------------------------------------------------------------------------------------------------------------------------

__global__ void calculateMeans(float *centroidsX_d, float *centroidsY_d, float *centroidsZ_d, float *sumsX_d, float *sumsY_d, float *sumsZ_d, int *numPoints_d){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < CLUSTER_NUM){
			centroidsX_d[tid] = sumsX_d[tid] / numPoints_d[tid];
			centroidsY_d[tid] = sumsY_d[tid] / numPoints_d[tid];
			centroidsZ_d[tid] = sumsZ_d[tid] / numPoints_d[tid];
	}
}


__global__ void kMeansKernel(float *pointsX_d, float *pointsY_d, float *pointsZ_d, float *centroidsX_d, float *centroidsY_d, float *centroidsZ_d, int *assignedCentroids_d, float *sumsX_d, float *sumsY_d, float *sumsZ_d, int *numPoints_d) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
		__shared__ float centroidsX_s[CLUSTER_NUM];
		__shared__ float centroidsY_s[CLUSTER_NUM];
		__shared__ float centroidsZ_s[CLUSTER_NUM];
		__shared__ float sumsX_s[CLUSTER_NUM];
		__shared__ float sumsY_s[CLUSTER_NUM];
		__shared__ float sumsZ_s[CLUSTER_NUM];
		__shared__ int numPoints_s[CLUSTER_NUM];

		if (threadIdx.x < CLUSTER_NUM) {
			centroidsX_s[threadIdx.x] = centroidsX_d[threadIdx.x];
			centroidsY_s[threadIdx.x] = centroidsY_d[threadIdx.x];
			centroidsZ_s[threadIdx.x] = centroidsZ_d[threadIdx.x];
			sumsX_s[threadIdx.x] = 0;
			sumsY_s[threadIdx.x] = 0;
			sumsZ_s[threadIdx.x] = 0;
			numPoints_s[threadIdx.x] = 0;
		}

		__syncthreads();

		if(tid < DATA_SIZE) {
			float clusterDistance = __FLT_MAX__;
			int oldCluster = assignedCentroids_d[tid];
			int currentCluster = oldCluster;
			float pX = pointsX_d[tid];
			float pY = pointsY_d[tid];
			float pZ = pointsZ_d[tid];


			for (int j = 0; j < CLUSTER_NUM; j++) {
				float distanceX = centroidsX_s[j] - pX;
				float distanceY = centroidsY_s[j] - pY;
				float distanceZ = centroidsZ_s[j] - pZ;
				float distance = sqrt(pow(distanceX, 2) + pow(distanceY, 2) + pow(distanceZ, 2));
				if (distance < clusterDistance) {
					clusterDistance = distance;
					currentCluster = j;
				}
			}

			//assign cluster and update partial sum
			assignedCentroids_d[tid] = currentCluster;
			atomicAdd(&sumsX_s[currentCluster], pX);
			atomicAdd(&sumsY_s[currentCluster], pY);
			atomicAdd(&sumsZ_s[currentCluster], pZ);
			atomicAdd(&numPoints_s[currentCluster], 1);

		}


		__syncthreads();

		//commit to global memory
		if(threadIdx.x < CLUSTER_NUM) {
			atomicAdd(&centroidsX_d[threadIdx.x], sumsX_s[threadIdx.x]);
			atomicAdd(&centroidsY_d[threadIdx.x], sumsY_s[threadIdx.x]);
			atomicAdd(&centroidsZ_d[threadIdx.x], sumsZ_s[threadIdx.x]);
			atomicAdd(&numPoints_d[threadIdx.x], numPoints_s[threadIdx.x]);
		}

}

__host__ void kMeansCuda(float *pointsX_h, float *pointsY_h, float *pointsZ_h){
	//device memory managing
	float *pointsX_d, *pointsY_d, *pointsZ_d;
	float *centroidsX_d, *centroidsY_d, *centroidsZ_d;
	float *sumPointsX_d, *sumPointsY_d, *sumPointsZ_d;
	int *assignedCentroids_d, *numPoints_d;
	int *assignedCentroids_h = (int*) malloc(sizeof(int) * DATA_SIZE);

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&pointsX_d, sizeof(float) * DATA_SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&pointsY_d, sizeof(float) * DATA_SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&pointsZ_d, sizeof(float) * DATA_SIZE));

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroidsX_d, sizeof(float) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroidsY_d, sizeof(float) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroidsZ_d, sizeof(float) * CLUSTER_NUM));

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&sumPointsX_d, sizeof(float) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&sumPointsY_d, sizeof(float) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&sumPointsZ_d, sizeof(float) * CLUSTER_NUM));

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&assignedCentroids_d, sizeof(int) * DATA_SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&numPoints_d, sizeof(int) * CLUSTER_NUM ));

	CUDA_CHECK_RETURN(cudaMemcpy(pointsX_d, pointsX_h, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(pointsY_d, pointsY_h, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(pointsZ_d, pointsZ_h, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice));

	// Step 1: Create k random centroids
	float *centroidsX_h = (float*) malloc(sizeof(float) * CLUSTER_NUM);
	float *centroidsY_h = (float*) malloc(sizeof(float) * CLUSTER_NUM);
	float *centroidsZ_h = (float*) malloc(sizeof(float) * CLUSTER_NUM);

	srand (time(NULL));
    vector<int> extractedIndex;

	for (int i = 0; i < CLUSTER_NUM; i++){
        bool alreadySelected = false;
        int randomIndex;
        do {                        //avoid repeating
        	randomIndex = rand() % DATA_SIZE - i;
        	for (int e : extractedIndex) {
        		if (randomIndex == e)
        	         alreadySelected = true;
        	}
        } while (alreadySelected);

        centroidsX_h[i] = pointsX_h[randomIndex];
		centroidsY_h[i] = pointsY_h[randomIndex];
		centroidsZ_h[i] = pointsZ_h[randomIndex];
	}

	CUDA_CHECK_RETURN(cudaMemcpy(centroidsX_d, centroidsX_h, sizeof(float) * CLUSTER_NUM, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(centroidsY_d, centroidsY_h, sizeof(float) * CLUSTER_NUM, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(centroidsZ_d, centroidsZ_h, sizeof(float) * CLUSTER_NUM, cudaMemcpyHostToDevice));

	for (int epoch = 0; epoch < MAX_ITERATIONS; epoch++) {
		CUDA_CHECK_RETURN(cudaMemset(numPoints_d, 0 , sizeof(int) * CLUSTER_NUM));
		CUDA_CHECK_RETURN(cudaMemset(sumPointsX_d, 0 , sizeof(float) * CLUSTER_NUM));
		CUDA_CHECK_RETURN(cudaMemset(sumPointsY_d, 0 , sizeof(float) * CLUSTER_NUM));
		CUDA_CHECK_RETURN(cudaMemset(sumPointsZ_d, 0 , sizeof(float) * CLUSTER_NUM));
		kMeansKernel<<<ceil((DATA_SIZE + 127) / 128), 128>>>(pointsX_d, pointsY_d, pointsZ_d, centroidsX_d, centroidsY_d, centroidsZ_d, assignedCentroids_d, sumPointsX_d, sumPointsY_d, sumPointsZ_d, numPoints_d); //TODO
		cudaDeviceSynchronize();
		calculateMeans<<<ceil((CLUSTER_NUM * 3 + 31) / 31), 32>>>(centroidsX_d, centroidsY_d, centroidsZ_d, sumPointsX_d, sumPointsY_d, sumPointsZ_d, numPoints_d);
		cudaDeviceSynchronize();
	}

	CUDA_CHECK_RETURN(cudaMemcpy(centroidsX_h, centroidsX_d, sizeof(float) * CLUSTER_NUM, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(centroidsY_h, centroidsY_d, sizeof(float) * CLUSTER_NUM, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(centroidsZ_h, centroidsZ_d, sizeof(float) * CLUSTER_NUM, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(assignedCentroids_h, assignedCentroids_d, sizeof(int) * DATA_SIZE, cudaMemcpyDeviceToHost));
	writeCsv(pointsX_h, pointsY_h, pointsZ_h, centroidsX_h, centroidsY_h, centroidsZ_h, assignedCentroids_h);

	 // Free host memory
	free(pointsX_h);
	free(pointsY_h);
	free(pointsZ_h);
	free(centroidsX_h);
	free(centroidsY_h);
	free(centroidsZ_h);
	free(assignedCentroids_h);

	 //free device memory
	cudaFree(pointsX_d);
	cudaFree(pointsY_d);
	cudaFree(pointsZ_d);
	cudaFree(centroidsX_d);
	cudaFree(centroidsY_d);
	cudaFree(centroidsZ_d);
	cudaFree(assignedCentroids_d);
	cudaFree(numPoints_d);
}

int main(int argc, char **argv){
	initialize();
	float *data = readCsv();

	float *dataX_h = (float*) malloc(sizeof(float) * DATA_SIZE);
	float *dataY_h = (float*) malloc(sizeof(float) * DATA_SIZE);
	float *dataZ_h = (float*) malloc(sizeof(float) * DATA_SIZE);

	for (int i = 0; i< DATA_SIZE; i++) {
		dataX_h[i] = data[i * 3];
		dataY_h[i] = data[i * 3 + 1];
		dataZ_h[i] = data[i * 3 + 2];
	}

	warm_up_gpu<<<128, 128>>>();  // avoids cold start for testing purposes
	auto start = high_resolution_clock::now();
	kMeansCuda(dataX_h, dataY_h, dataZ_h);
	auto end = high_resolution_clock::now();

	auto ms_int = duration_cast<milliseconds>(end - start);
	cout << "duration in milliseconds: " << ms_int.count() <<"\n";

    return ms_int.count();


}

