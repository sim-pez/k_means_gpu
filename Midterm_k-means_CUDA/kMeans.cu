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

__global__ void warmUpGpu(){  // this kernel avoids cold start when evaluating duration of kmeans exec.
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


__global__ void kMeansKernel(float *pointsX_d, float *pointsY_d, float *pointsZ_d, float *centroidsX_d, float *centroidsY_d, float *centroidsZ_d, int *assignedCentroids_d, float *sumsX_d, float *sumsY_d, float *sumsZ_d, int *numPoints_d, int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float centroidsX_s[CLUSTER_NUM];
	__shared__ float centroidsY_s[CLUSTER_NUM];
	__shared__ float centroidsZ_s[CLUSTER_NUM];
	__shared__ float sumsX_s[CLUSTER_NUM];
	__shared__ float sumsY_s[CLUSTER_NUM];
	__shared__ float sumsZ_s[CLUSTER_NUM];
	__shared__ int numPoints_s[CLUSTER_NUM];

	int tx = threadIdx.x;
	if (tx < CLUSTER_NUM) {
		centroidsX_s[tx] = centroidsX_d[tx];
		centroidsY_s[tx] = centroidsY_d[tx];
		centroidsZ_s[tx] = centroidsZ_d[tx];
		sumsX_s[tx] = 0;
		sumsY_s[tx] = 0;
		sumsZ_s[tx] = 0;
		numPoints_s[tx] = 0;
	}

	__syncthreads();

	if(tid < n) {
		float clusterDistance = __FLT_MAX__;
		int currentCluster = assignedCentroids_d[tid];
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
	if(tx < CLUSTER_NUM) {
		atomicAdd(&sumsX_d[tx], sumsX_s[tx]);
		atomicAdd(&sumsY_d[tx], sumsY_s[tx]);
		atomicAdd(&sumsZ_d[tx], sumsZ_s[tx]);
		atomicAdd(&numPoints_d[tx], numPoints_s[tx]);
	}

}

__host__ void kMeansCuda(float *pointsX_h, float *pointsY_h, float *pointsZ_h, int n){
	//device memory managing
	float *pointsX_d, *pointsY_d, *pointsZ_d;
	float *centroidsX_d, *centroidsY_d, *centroidsZ_d;
	float *sumPointsX_d, *sumPointsY_d, *sumPointsZ_d;
	int *assignedCentroids_d, *numPoints_d;
	int *assignedCentroids_h = (int*) malloc(sizeof(int) * n);

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&pointsX_d, sizeof(float) * n));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&pointsY_d, sizeof(float) * n));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&pointsZ_d, sizeof(float) * n));

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroidsX_d, sizeof(float) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroidsY_d, sizeof(float) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroidsZ_d, sizeof(float) * CLUSTER_NUM));

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&sumPointsX_d, sizeof(float) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&sumPointsY_d, sizeof(float) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&sumPointsZ_d, sizeof(float) * CLUSTER_NUM));

	CUDA_CHECK_RETURN(cudaMalloc((void ** )&assignedCentroids_d, sizeof(int) * n));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&numPoints_d, sizeof(int) * CLUSTER_NUM ));

	CUDA_CHECK_RETURN(cudaMemcpy(pointsX_d, pointsX_h, sizeof(float) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(pointsY_d, pointsY_h, sizeof(float) * n, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(pointsZ_d, pointsZ_h, sizeof(float) * n, cudaMemcpyHostToDevice));

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
        		randomIndex = rand() % n - i;
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

	for (int epoch = 0; epoch < ITERATIONS; epoch++) {
		CUDA_CHECK_RETURN(cudaMemset(numPoints_d, 0 , sizeof(int) * CLUSTER_NUM));
		CUDA_CHECK_RETURN(cudaMemset(sumPointsX_d, 0 , sizeof(float) * CLUSTER_NUM));
		CUDA_CHECK_RETURN(cudaMemset(sumPointsY_d, 0 , sizeof(float) * CLUSTER_NUM));
		CUDA_CHECK_RETURN(cudaMemset(sumPointsZ_d, 0 , sizeof(float) * CLUSTER_NUM));
		kMeansKernel<<<((n + 127) / 128), 128>>>(pointsX_d, pointsY_d, pointsZ_d, centroidsX_d, centroidsY_d, centroidsZ_d, assignedCentroids_d, sumPointsX_d, sumPointsY_d, sumPointsZ_d, numPoints_d, n);
		cudaDeviceSynchronize();
		calculateMeans<<<((CLUSTER_NUM * 3 + 31) / 32), 32>>>(centroidsX_d, centroidsY_d, centroidsZ_d, sumPointsX_d, sumPointsY_d, sumPointsZ_d, numPoints_d);
		cudaDeviceSynchronize();
	}

	CUDA_CHECK_RETURN(cudaMemcpy(centroidsX_h, centroidsX_d, sizeof(float) * CLUSTER_NUM, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(centroidsY_h, centroidsY_d, sizeof(float) * CLUSTER_NUM, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(centroidsZ_h, centroidsZ_d, sizeof(float) * CLUSTER_NUM, cudaMemcpyDeviceToHost));
	CUDA_CHECK_RETURN(cudaMemcpy(assignedCentroids_h, assignedCentroids_d, sizeof(int) * n, cudaMemcpyDeviceToHost));
	writeCsv(pointsX_h, pointsY_h, pointsZ_h, centroidsX_h, centroidsY_h, centroidsZ_h, assignedCentroids_h, n);

	
	free(pointsX_h);
	free(pointsY_h);
	free(pointsZ_h);
	free(centroidsX_h);
	free(centroidsY_h);
	free(centroidsZ_h);
	free(assignedCentroids_h);
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
	int n = 1000; //number of points

	initialize();
	float *data = readCsv();

	float *dataX_h = (float*) malloc(sizeof(float) * n);
	float *dataY_h = (float*) malloc(sizeof(float) * n);
	float *dataZ_h = (float*) malloc(sizeof(float) * n);

	for (int i = 0; i< n; i++) {
		dataX_h[i] = data[i * 3];
		dataY_h[i] = data[i * 3 + 1];
		dataZ_h[i] = data[i * 3 + 2];
	}

	warmUpGpu<<<128, 128>>>();  // avoids cold start for testing purposes
	auto start = high_resolution_clock::now();
	kMeansCuda(dataX_h, dataY_h, dataZ_h, n);
	auto end = high_resolution_clock::now();

	auto ms_int = duration_cast<milliseconds>(end - start);
	cout << "duration in milliseconds: " << ms_int.count() <<"\n";

	return ms_int.count();


}

