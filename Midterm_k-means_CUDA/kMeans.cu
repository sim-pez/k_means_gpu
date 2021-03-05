#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <cuda.h>
#include <random>

#include "operations.h"
#include "Point.h"
#include "csvHandler.h"
#include "definitions.h"

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
__global__ void updateCentroids(float *points_d, float *centroids_d, int *assignedCentroids_d, int *numPoints_d, float *sums_d){ //float *sums_d

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float sums_s[CLUSTER_NUM * 3]; //TODO need atomicAdd()
	__shared__ int numPoints_s[CLUSTER_NUM]; //TODO better unsigned int
	if(threadIdx.x < CLUSTER_NUM * 3) {
		if(threadIdx.x < CLUSTER_NUM) {
			numPoints_s[threadIdx.x] = 0;
		}
		sums_s[threadIdx.x] = 0.0f;
	}
	__syncthreads();


	if (tid < DATA_SIZE){
		int cluster = assignedCentroids_d[tid];
		sums_s[cluster * 3] += points_d[tid * 3];
		sums_s[cluster * 3 + 1] += points_d[tid * 3 + 1];
		sums_s[cluster * 3 + 2] += points_d[tid * 3 + 2];
		numPoints_s[cluster]++;
	}

	__syncthreads();

	if(threadIdx.x < CLUSTER_NUM * 3) {
		atomicAdd(&sums_d[threadIdx.x], sums_s[threadIdx.x]);
		if(threadIdx.x < CLUSTER_NUM) {
			atomicAdd(&numPoints_d[threadIdx.x], numPoints_s[threadIdx.x]);
		}
	}
	__syncthreads(); //FIXME

	if (tid < CLUSTER_NUM * 3){
		int newValue = sums_d[tid] / numPoints_d[tid / 3];
		centroids_d[tid] = newValue;
	}
}

__global__ void assignClusters(float *points_d, float *centroids_d, int *assignedCentroids_d, bool *clusterChanged){

	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < DATA_SIZE){
    	float pX = points_d[tid * 3];
    	float pY = points_d[tid * 3 + 1];
    	float pZ = points_d[tid * 3 + 2];
        float clusterDistance = __FLT_MAX__;
        int oldCluster = assignedCentroids_d[tid];
        int currentCluster = oldCluster;

        for (int j = 0; j < CLUSTER_NUM; j++) {
        	int distanceX = centroids_d[j * 3] - pX;
        	int distanceY = centroids_d[j * 3 + 1] - pY;
        	int distanceZ = centroids_d[j * 3 + 2] - pZ;
            float distance = sqrt(pow(distanceX , 2) + pow(distanceY , 2) + pow(distanceZ , 2));
            if (distance < clusterDistance) {
                clusterDistance = distance;
                oldCluster = currentCluster;
                currentCluster = j;
            }
        }

        if (currentCluster != oldCluster) {
            *clusterChanged = true;
            assignedCentroids_d[tid] = currentCluster;
        }
    }
}

__host__ void kMeansCuda(float *points_h, int epochsLimit){
	//device memory managing
	float *points_d, *centroids_d, *sums_d;
	int *assignedCentroids_d, * clusterSize_d,  *numPoints_d;
	int *assignedCentroids_h = (int*) malloc(sizeof(int) * DATA_SIZE);
	int *clusterSize_h = (int*) malloc(sizeof(int) * CLUSTER_NUM);
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&points_d, sizeof(float) * DATA_SIZE * 3));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroids_d, sizeof(float) * CLUSTER_NUM * 3));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&assignedCentroids_d, sizeof(int) * DATA_SIZE));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&clusterSize_d, sizeof(int) * CLUSTER_NUM));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&numPoints_d, sizeof(int) * CLUSTER_NUM ));
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&sums_d, sizeof(float) * CLUSTER_NUM * 3));

	CUDA_CHECK_RETURN(cudaMemcpy(points_d, points_h, sizeof(float) * DATA_SIZE * 3, cudaMemcpyHostToDevice)); // TODO copy in constant memory


	// Step 1: Create k random centroids
	printf("Step1 \n");
	float *centroids_h = (float*) malloc(sizeof(float) * CLUSTER_NUM * 3);
	srand(time(NULL));
	int randNum = rand() % (DATA_SIZE / CLUSTER_NUM);
	for (int i = 0; i < CLUSTER_NUM; i++){
		int randomLocation = randNum + DATA_SIZE*i/CLUSTER_NUM;
		centroids_h[i * 3] = points_h[randomLocation  * 3];
		centroids_h[i * 3 + 1] = points_h[randomLocation * 3 + 1];
		centroids_h[i * 3 + 2] = points_h[randomLocation * 3 + 2];
		clusterSize_h[i] = 0;
	}

	CUDA_CHECK_RETURN(cudaMemcpy(centroids_d, centroids_h, sizeof(float) * CLUSTER_NUM * 3, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(clusterSize_d, clusterSize_h, sizeof(int) * CLUSTER_NUM, cudaMemcpyHostToDevice));

	bool clusterChanged_h = false;
	bool *ptrCgd_h = &clusterChanged_h;
	bool *clusterChanged_d;
	CUDA_CHECK_RETURN(cudaMalloc(&clusterChanged_d, sizeof(bool)));

	int epoch = 0;
	while(epoch < epochsLimit) {
	    writeCsv(points_h, centroids_h, assignedCentroids_h, epoch);
	    //Step 2: assign dataPoints to the clusters, based on the distance from its centroid

	    printf("Step2 \n");
	    CUDA_CHECK_RETURN(cudaMemcpy(clusterChanged_d, ptrCgd_h, sizeof(bool), cudaMemcpyHostToDevice));
	    CUDA_CHECK_RETURN(cudaMemset(numPoints_d, 0 , sizeof(int) * CLUSTER_NUM));
	    CUDA_CHECK_RETURN(cudaMemset(sums_d, 0 , sizeof(float) * CLUSTER_NUM * 3));
	    assignClusters<<<(DATA_SIZE + 127)/ 128 , 128>>>(points_d, centroids_d, assignedCentroids_d, clusterChanged_d); //why + 127?

	    cudaDeviceSynchronize();
	    CUDA_CHECK_RETURN(cudaMemcpy(assignedCentroids_h, assignedCentroids_d, sizeof(int) * DATA_SIZE, cudaMemcpyDeviceToHost));

	    CUDA_CHECK_RETURN(cudaMemcpy(ptrCgd_h, clusterChanged_d, sizeof(bool), cudaMemcpyDeviceToHost));
	    if (!clusterChanged_h) {
	        printf("Nothing changed...exiting \n");
	    	break;          // exit if clusters has not been changed
	    }
	    else {
	    	clusterChanged_h = false;
	    }

	    //Step 3: update centroids

	    printf("Step3 \n");
	    updateCentroids<<<(DATA_SIZE + 127)/ 128 , 128>>>(points_d, centroids_d, assignedCentroids_d, numPoints_d, sums_d);
	    cudaDeviceSynchronize();
	    printf("finished updating centroids \n");
	    CUDA_CHECK_RETURN(cudaMemcpy(centroids_d, centroids_h, sizeof(float) * CLUSTER_NUM * 3, cudaMemcpyHostToDevice));
	    epoch++;
	    printf("iteration completed, starting a new one... \n");
	 }

	CUDA_CHECK_RETURN(cudaMemcpy(centroids_d, centroids_h, sizeof(float) * CLUSTER_NUM * 3, cudaMemcpyHostToDevice));

	writeCsv(points_h, centroids_h, assignedCentroids_h, __INT_MAX__);
	if (epoch == epochsLimit){
	    cout << "Maximum number of iterations reached!";
	}
	cout << "iterations = " << epoch << "\n";

	 // Free host memory //TODO check if they are correct
	free(points_h);
	free(centroids_h);
	free(assignedCentroids_h);

	 //free device memory
	cudaFree(points_d);
	cudaFree(centroids_d);
	cudaFree(assignedCentroids_d);
	cudaFree(clusterChanged_d);

}

int main(int argc, char **argv)
{

	int maxIterations = 500;
	initialize();
	float *data_h = readCsv();
	kMeansCuda(data_h, maxIterations);

}
