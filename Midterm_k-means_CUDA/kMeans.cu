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
void updateCentroids(float *points_h, float *centroids_h, float  *assignedCentroids_h, int k){

    int numPoints[k];
    float sumX[k] = {0};
    float sumY[k] = {0};
    float sumZ[k]= {0};

    for (int i = 0; i < DATA_SIZE; i++) {
        int pointCluster = assignedCentroids_h[i];
        sumX[pointCluster] += points_h[i * 3];
        sumY[pointCluster] += points_h[i * 3 + 1];
        sumZ[pointCluster] += points_h[i * 3 + 2];
        numPoints[pointCluster]++;
    }

    for (int i = 0; i < k; i++){
        float newX = sumX[i] / numPoints[i];
        float newY = sumY[i] / numPoints[i];
        float newZ = sumZ[i] / numPoints[i];

        centroids_h[i * 3] = newX;
        centroids_h[i * 3 + 1] = newY;
        centroids_h[i * 3 + 2] = newZ;
    }

}


__global__ void assignClusters(float *points_d, float *centroids_d, float *assignedCentroids_d, int k, bool *clusterChanged){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < DATA_SIZE){
    	int pX = points_d[tid * 3];
    	int pY = points_d[tid * 3 + 1];
    	int pZ = points_d[tid * 3 + 2];

        float clusterDistance = __FLT_MAX__;
        int oldCluster = assignedCentroids_d[tid];
        int currentCluster = oldCluster;

        for (int j = 0; j < k; j++) {
        	int distanceX = centroids_d[j * 3] - pX;
        	int distanceY = centroids_d[j * 3 + 1] - pY;
        	int distanceZ = centroids_d[j * 3 + 2] - pZ;
            float distance = sqrt(pow(distanceX , 2) + pow(distanceY , 2) + pow(distanceZ , 2));
            if (distance < clusterDistance) {
                clusterDistance = distance;
                oldCluster = currentCluster;
                currentCluster = k;
            }
        }

        if (currentCluster != oldCluster) {
            *clusterChanged = true;
            assignedCentroids_d[tid] = currentCluster;
        }
    }

}

__host__ void kMeansCuda(float *points_h, int epochsLimit, int k){
	// Step 1: Create k random centroids
	float *centroids_h = (float*) malloc(sizeof(float) * k * 3);
	srand (time(NULL));
	int randNum = rand() % (DATA_SIZE / k);
	for (int i = 0; i < k; i++){
		int randomLocation = randNum + DATA_SIZE*i/k;
		centroids_h[i * 3] = points_h[randomLocation  * 3];
		centroids_h[i * 3 + 1] = points_h[randomLocation * 3 + 1];
		centroids_h[i * 3 + 2] = points_h[randomLocation * 3 + 2];
	}

	//device memory managing
	float *assignedCentroids_h = (float*) malloc(sizeof(float) * DATA_SIZE);
	float *points_d, *centroids_d, *assignedCentroids_d;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&points_d, sizeof(float) * DATA_SIZE * 3)); // allocate device memory
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroids_d, sizeof(float) * k * 3));
	CUDA_CHECK_RETURN(cudaMalloc((void**)&assignedCentroids_d, sizeof(float) * DATA_SIZE));

	CUDA_CHECK_RETURN(cudaMemcpy(points_d, points_h, sizeof(float) * DATA_SIZE * 3, cudaMemcpyHostToDevice)); // TODO copy in constant memory
	CUDA_CHECK_RETURN(cudaMemcpy(centroids_d, centroids_h, sizeof(float) * k * 3, cudaMemcpyHostToDevice));
	CUDA_CHECK_RETURN(cudaMemcpy(assignedCentroids_d, assignedCentroids_h, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice));

	int epoch = 0;
	while(epoch < epochsLimit) {
	    writeCsv(points_h, centroids_h, assignedCentroids_h, epoch, k);

	    //Step 2: assign dataPoints to the clusters, based on the distance from its centroid
	    bool clusterChanged_h = false;
	    bool *ptrCgd_h = &clusterChanged_h;
	    bool *clusterChanged_d;
	    CUDA_CHECK_RETURN(cudaMalloc(&clusterChanged_d, sizeof(bool)));
	    CUDA_CHECK_RETURN(cudaMemcpy(clusterChanged_d, ptrCgd_h, sizeof(bool), cudaMemcpyHostToDevice));
	    assignClusters<<<(DATA_SIZE + 127)/ 128 , 128>>>(points_d, centroids_d, assignedCentroids_d, k, clusterChanged_d); //why + 127?
	    CUDA_CHECK_RETURN(cudaMemcpy(ptrCgd_h, clusterChanged_d, sizeof(bool), cudaMemcpyDeviceToHost));
	    cudaDeviceSynchronize();


	    float *assignedCentroids_h;
	    CUDA_CHECK_RETURN(cudaMemcpy(assignedCentroids_h, assignedCentroids_d, sizeof(float) * DATA_SIZE, cudaMemcpyDeviceToHost));


	    if (!clusterChanged_h) {
	        break;          // exit if clusters has not been changed
	    }

	    //Step 3: update centroids
	    updateCentroids(points_h, centroids_h, assignedCentroids_h, k);
	    CUDA_CHECK_RETURN(cudaMemcpy(centroids_d, centroids_h, sizeof(Point) * k, cudaMemcpyHostToDevice));

	    epoch++;
	 }

	CUDA_CHECK_RETURN(cudaMemcpy(centroids_d, centroids_h, sizeof(Point) * k, cudaMemcpyHostToDevice));

	 writeCsv(points_h, centroids_h, assignedCentroids_h, __INT_MAX__, k);
	 if (epoch == epochsLimit){
	     cout << "Maximum number of iterations reached!";
	 }
	 cout << "iterations = " << epoch << "\n";

	 // Free host memory
	 free(points_h);
	 //free GPU memory
	 cudaFree(points_h); //TODO add free() and cudaFree()

}


int main(int argc, char **argv)
{
	int maxIterations = 500;
	initialize();
	float *data_h = readCsv();
	kMeansCuda(data_h, maxIterations, CLUSTER_NUM);
}
