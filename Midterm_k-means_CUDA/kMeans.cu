#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <numeric>
#include <cuda.h>
#include <random>
#include <vector>


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

    vector<int> numPoints(k, 0);
    vector<float> sumX(k, 0.0);
    vector<float> sumY(k, 0.0);
    vector<float> sumZ(k, 0.0);


    for (int i = 0; i< DATA_SIZE; i++) {
        int pointCluster = assignedCentroids_h->i;
        int xP = i * 3;
        int yP = i * 3 + 1;
        int zP = i * 3 + 2;
        sumX.at(pointCluster) += points_h->xP;
        sumY.at(pointCluster) += points_h->yP;
        sumZ.at(pointCluster) += points_h->zP;
        numPoints.at(pointCluster)++;
    }

    for (int i = 0; i < k; i++){
        float newX = sumX.at(i) / numPoints.at(i);
        float newY = sumY.at(i) / numPoints.at(i);
        float newZ = sumZ.at(i) / numPoints.at(i);

        int xC = i * 3;
        int yC = i * 3 + 1;
        int zC = i * 3 + 2;

        centroids_h->xC = newX;
        centroids_h->yC = newY;
        centroids_h->zC = newZ;
    }

}


__device__ void assignClusters(float *points_d, float *centroids_d, float *assignedCentroids_d, int k, bool *clusterChanged){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < DATA_SIZE){
    	int xP = tid * 3;
    	int yP = tid * 3 + 1;
    	int zP = tid * 3 + 2;


        float clusterDistance = FLT_MAX;  // the distance between its actual cluster' centroid
        int oldCluster = assignedCentroids_d->tid; // keep trace of witch cluster the point is
        int currentCluster = oldCluster; // at the beginning the actual cluster coincide with the old one

        for (int j = 0; j < k; j++) {

        	int xC = k * 3;
        	int yC = k * 3 +1;
        	int zC = k * 3 +2;

            float distance = distance3d(points_d->xP, points_d->yP, points_d->zP, centroids_d->xC, centroids_d->yC, centroids_d->zC);
            if (distance < clusterDistance) {
                clusterDistance = distance;
                oldCluster = currentCluster;
                currentCluster = k;
            }
        }

        if (currentCluster != oldCluster()) {
            *clustersChanged = true;
            assignedCentroids_d->Idx = currentCluster;
        }
    }

}

__host__ void kMeansCuda(float *points_h, int epochsLimit, int k){
	// Step 1: Create k random centroids
	float *centroids_h = (float*) malloc(sizeof(float) * k * 3);
	random_device rd;
	default_random_engine engine(rd());
	uniform_int_distribution<int> distribution(0, DATA_SIZE - 1);
	int lastEpoch = 0;
	for(int i=0; i<k; i++) {
		int randomLocation = distribution(engine) % 3; // it is a random x axis coordinate
	    int xP = randomLocation;        // getting the coordinates location  of the random point choosen as centroid
	    int yP = randomLocation + 1;
	    int zP = randomlocation * 2;

	    int xC = k * 3;                // getting the coordinates location of the centroid
	    int yC = k * 3 + 1;
	    int zC = k * 3 + 2;

	    centroids_h->xC = points_h->xP;   // assigning centroid' coordinates
	    centroids_h->yC = points_h->yP;
	    centroids_h->zC = points_h->zP;
	}


	//device memory managing
	__device__ __constant__ float *points_d;
	__device__ float *centroids_d;
	__device__ float *assignedCentroids_d;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&points_d, sizeof(float) * DATA_SIZE * 3)); // allocate device memory
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroids_d, sizeof(float) * k * 3));
	CUDA_CHECK_RETURN(cudaMAlloc((void**)&assignedCentroids_d, sizeof(float) * DATA_SIZE));

	CUDA_CHECK_RETURN(cudaMemcpyToSimbol(points_h, points_d, sizeof(float) * DATA_SIZE));
	CUDA_CHECK_RETURN(cudaMemcpy(centroids_h, centroids_d, sizeof(float) * DATA_SIZE, cudaMemcpyHostToDevice));



	//TODO shall I allocate something to constant memory ?

	for(int ep = 0 ; ep < epochsLimit; ep++) {

	    writeCsv(points_h, &centroids_h, ep, k);


	    //Step 2: assign dataPoints to the clusters, based on the distance from its centroid
	    __device__ bool clusterChanged = false;
	    assignClusters<<(DATA_SIZE + 127) / 128 , 128>>(points_d, centroids_d, assignedCentroids_d, k, &clusterChanged); //why + 127?
	    cudaDeviceSynchronize();

	    CUDA_CHECK_RETURN(cudaMemcpy(assignedCentroids_d, assignedCentroids_h, sizeof(float) * DATA_SIZE, cudaMemcpyDeviceToHost));


	    if (!clustersChanged) {
	        break;          // exit if clusters has not been changed
	    }

	    //Step 3: update centroids
	    updateCentroids(points_h, centroids_h, assignedCentroids_h, k);
	    CUDA_CHECK_RETURN(cudaMemcpy(centroids_h, centroids_d, sizeof(Point) * k, cudaMemcpyHostToDevice));

	    lastEpoch = ep;
	 }

	CUDA_CHECK_RETURN(cudaMemcpy(centroids_h, centroids_d, sizeof(Point) * k, cudaMemcpyHostToDevice));

	 writeCsv(points_h, &centroids_h, __INT_MAX__, k);
	 if (lastEpoch == epochsLimit){
	     cout << "Maximum number of iterations reached!";
	 }
	 cout << "iterations = " << lastEpoch << "\n";

	 // Free host memory
	 free(points_h);
	 //free GPU memory
	 cudaFree(points_h);

}


int main(int argc, char **argv)
{
	int maxIterations = 500;
	initialize();
	vector<Point> data_h = readCsv();
	kMeansCuda(&data_h, 500, CLUSTER_NUM);
}
