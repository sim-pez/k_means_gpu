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

__global__ void updateCentroids(float *points_d, float *centroids_d, int *assignedCentroids_d, int *numPoints_d, int numDataset){
 // more parallelizable
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float sums_s[CLUSTER_NUM * 3];
	__shared__ int numPoints_s[CLUSTER_NUM];
	if(threadIdx.x < CLUSTER_NUM * 3) {
		if(threadIdx.x < CLUSTER_NUM) {
			numPoints_s[threadIdx.x] = 0;
		}
		sums_s[threadIdx.x] = 0.0f;
	}
	__syncthreads();

	if (tid < DATA_SIZE * numDataset){
		int cluster = assignedCentroids_d[tid];
		atomicAdd(&sums_s[cluster * 3], points_d[tid * 3]);
		atomicAdd(&sums_s[cluster * 3 + 1], points_d[tid * 3 + 1]);
		atomicAdd(&sums_s[cluster * 3 + 2], points_d[tid * 3 + 2]);
		atomicAdd(&numPoints_s[cluster], 1);
	}

	__syncthreads();

	//commit to global memory
	if(threadIdx.x < CLUSTER_NUM * 3) {
		atomicAdd(&centroids_d[threadIdx.x], sums_s[threadIdx.x]);
		if(threadIdx.x < CLUSTER_NUM) {
			atomicAdd(&numPoints_d[threadIdx.x], numPoints_s[threadIdx.x]);
		}
	}

}

__global__ void calculateMeans(float *centroids_d, int *numPoints_d){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if(tid < CLUSTER_NUM * 3){
			centroids_d[tid] = centroids_d[tid] / numPoints_d[tid / 3];
	}
}


__global__ void assignClusters(float *points_d, float *centroids_d, int *assignedCentroids_d, bool *clusterChanged, int numDataset){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ float centroids_s[CLUSTER_NUM * 3];
	if (threadIdx.x < CLUSTER_NUM * 3) {
		centroids_s[threadIdx.x] = centroids_d[threadIdx.x];
	}

	if(tid < DATA_SIZE * numDataset) {
		float clusterDistance = __FLT_MAX__;
		int oldCluster = assignedCentroids_d[tid];
		int currentCluster = oldCluster;
		float pX = points_d[tid * 3];
		float pY = points_d[tid * 3 + 1];
		float pZ = points_d[tid * 3 + 2];


		for (int j = 0; j < CLUSTER_NUM; j++) {
			float distanceX = centroids_s[j * 3] - pX;
			float distanceY = centroids_s[j * 3 + 1] - pY;
			float distanceZ = centroids_s[j * 3 + 2] - pZ;
			float distance = sqrt(pow(distanceX, 2) + pow(distanceY, 2) + pow(distanceZ, 2));
			if (distance < clusterDistance) {
				clusterDistance = distance;
				currentCluster = j;
			}
		}

		if (currentCluster != oldCluster) {
		   *clusterChanged = true;
		   assignedCentroids_d[tid] = currentCluster;
		}
	}

}

__host__ void kMeansCuda(float *points_h, int epochsLimit, int numDataset){
	//device memory managing
	float *points_d, *centroids_d;
	int *assignedCentroids_d, *numPoints_d;
	int *assignedCentroids_h = (int*) malloc(sizeof(int) * DATA_SIZE * numDataset);
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&points_d, sizeof(float) * DATA_SIZE * numDataset * 3));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroids_d, sizeof(float) * CLUSTER_NUM * 3));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&assignedCentroids_d, sizeof(int) * DATA_SIZE * numDataset));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&numPoints_d, sizeof(int) * CLUSTER_NUM ));

	CUDA_CHECK_RETURN(cudaMemcpy(points_d, points_h, sizeof(float) * DATA_SIZE * numDataset * 3, cudaMemcpyHostToDevice)); 

	// Step 1: Create k random centroids
	float *centroids_h = (float*) malloc(sizeof(float) * CLUSTER_NUM * 3); 
	//srand(time(NULL));
	//int randNum = 5;
	//int randNum = rand() % ((DATA_SIZE * numDataset) / CLUSTER_NUM);
	random_device rd;
	default_random_engine engine(rd());
	uniform_int_distribution<int> distribution(0, DATA_SIZE * numDataset - 1);
	for (int i = 0; i < CLUSTER_NUM; i++){
		int randomLocation = distribution(engine);
		//int randomLocation = randNum + (DATA_SIZE * numDataset) * i / CLUSTER_NUM;
		centroids_h[i * 3] = points_h[randomLocation  * 3];
		centroids_h[i * 3 + 1] = points_h[randomLocation * 3 + 1];
		centroids_h[i * 3 + 2] = points_h[randomLocation * 3 + 2];
	}

	CUDA_CHECK_RETURN(cudaMemcpy(centroids_d, centroids_h, sizeof(float) * CLUSTER_NUM * 3, cudaMemcpyHostToDevice));

	bool clusterChanged_h = false;
	bool *ptrCgd_h = &clusterChanged_h;
	bool *clusterChanged_d;
	CUDA_CHECK_RETURN(cudaMalloc(&clusterChanged_d, sizeof(bool)));



	int epoch = 0;
	while(epoch < epochsLimit) {
		//Step 2: assign dataPoints to the clusters, based on the distance from its centroid

		CUDA_CHECK_RETURN(cudaMemcpy(clusterChanged_d, ptrCgd_h, sizeof(bool), cudaMemcpyHostToDevice));
		assignClusters<<<(DATA_SIZE * numDataset + 127)/ 128 , 128>>>(points_d, centroids_d, assignedCentroids_d, clusterChanged_d, numDataset);
		cudaDeviceSynchronize();

		//write a csv file at each iteration to check how k-means is assigning clusters
		//CUDA_CHECK_RETURN(cudaMemcpy(assignedCentroids_h, assignedCentroids_d, sizeof(int) * DATA_SIZE, cudaMemcpyDeviceToHost));
		CUDA_CHECK_RETURN(cudaMemcpy(ptrCgd_h, clusterChanged_d, sizeof(bool), cudaMemcpyDeviceToHost));
		//writeCsv(points_h, centroids_h, assignedCentroids_h, epoch);


		if (!clusterChanged_h) {
		    //printf("Nothing changed...exiting \n");
		    break;          // exit if clusters has not been changed
		}
		else { clusterChanged_h = false; }

		//Step 3: update centroids

		// set numPoints_d and centroids_d to 0 so updateCentroids can do is stuff to evaluate the new position of the centroids
		CUDA_CHECK_RETURN(cudaMemset(numPoints_d, 0 , sizeof(int) * CLUSTER_NUM));
		CUDA_CHECK_RETURN(cudaMemset(centroids_d, 0 , sizeof(float) * CLUSTER_NUM * 3));
		updateCentroids<<<(DATA_SIZE * numDataset + 127) / 128 , 128>>>(points_d, centroids_d, assignedCentroids_d, numPoints_d, numDataset);
		cudaDeviceSynchronize();
		calculateMeans<<<(CLUSTER_NUM * 3 + 31) / 32, 32 >>>(centroids_d, numPoints_d);
		cudaDeviceSynchronize();
		//CUDA_CHECK_RETURN(cudaMemcpy(centroids_h, centroids_d, sizeof(float) * CLUSTER_NUM * 3, cudaMemcpyDeviceToHost));  // use it in case you want the code to write the csv's at each iteration


		//printf("iteration %d complete\n", epoch + 1);
		epoch++;
	}


	if (epoch == epochsLimit){
		printf("Maximum number of iterations reached! \n");
	}

	printf("iterations = %d \n", epoch);

	 // Free host memory
	//free(points_h);
	free(centroids_h);
	free(assignedCentroids_h);

	 //free device memory
	cudaFree(points_d);
	cudaFree(centroids_d);
	cudaFree(assignedCentroids_d);
	cudaFree(clusterChanged_d);
	cudaFree(numPoints_d);
}

int main(int argc, char **argv){

	initialize();
	float *data_h = readCsv(1);
	warm_up_gpu<<<128, 128>>>();  // avoiding cold start...
	auto start = high_resolution_clock::now();
	kMeansCuda(data_h, MAX_ITERATIONS, 1);
	auto end = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(end - start);
	cout<< "duration = " << duration.count() << " microseconds" << endl;

	free(data_h);

}

