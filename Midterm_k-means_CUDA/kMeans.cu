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
void updateCentroids(vector<Point> *points, vector<Point> *centroids, int k){      //TODO parallelizable?

    vector<int> numPoints(k, 0);
    vector<float> sumX(k, 0.0);
    vector<float> sumY(k, 0.0);
    vector<float> sumZ(k, 0.0);

    if (tid < DATA_SIZE){
    	points->at();
    }
    for (auto &point : *points) {
        int pointCluster = point.getCluster();
        sumX.at(pointCluster) += point.getX();
        sumY.at(pointCluster) += point.getY();
        sumZ.at(pointCluster) += point.getZ();
        numPoints.at(pointCluster)++;
    }

    for (int i = 0; i < k; i++){
        float newX = sumX.at(i) / numPoints.at(i);
        float newY = sumY.at(i) / numPoints.at(i);
        float newZ = sumZ.at(i) / numPoints.at(i);

        centroids->at(i).setX(newX);
        centroids->at(i).setY(newY);
        centroids->at(i).setZ(newZ);
    }

}

// TODO clusterChanged can be saved using atomicOR() ?
__device__ void assignClusters(vector<Point> *points, vector<Point> *centroids, int k, bool *clusterChanged){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < DATA_SIZE){
    	Point point = points->at(tid);
        point.setClusterDistance(__DBL_MAX__);
        point.setOldCluster(point.getCluster());

        float clusterDistance = point.getClusterDistance();  // the distance between its actual cluster' centroid
        int clusterIndex = point.getCluster(); // keep trace of witch cluster the point is

        for (int j = 0; j < k; j++) {
            float distance = distance3d(centroids->at(j), point);
            if (distance < clusterDistance) {
                point.setClusterDistance(distance);
                clusterDistance = distance;
                point.setCluster(j);
                clusterIndex = j;
            }
        }

        if (point.getCluster() != point.getOldCluster()) {
            *clustersChanged = true;
        }
    }

}

__host__ void kMeansCuda(vector<Point> *points_h, int epochsLimit, int k){
	// Step 1: Create k random centroids
	vector<Point> centroids_h;
	random_device rd;
	default_random_engine engine(rd());
	uniform_int_distribution<int> distribution(0, points_h->size() - 1);
	int lastEpoch = 0;
	for(int i=0; i<k; i++) {
	    int randomLocation = distribution(engine);
	    Point c = points_h->at(randomLocation);
	    centroids_h.push_back(c);
	}


	//device memory managing
	vector<Point> *points_d, centroids_d;
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&points_d, sizeof(Point) * DATA_SIZE)); // allocate device memory
	CUDA_CHECK_RETURN(cudaMalloc((void ** )&centroids_d, sizeof(Point) * DATA_SIZE));


	//TODO shall I allocate something to constant memory ?

	for(int ep = 0 ; ep < epochsLimit; ep++) {

	    writeCsv(points_h, &centroids_h, ep, k);


	    //Step 2: assign dataPoints to the clusters, based on the distance from its centroid
	    __device__ bool clusterChanged = false;
	    assignClusters<<(DATA_SIZE + 127) / 128 , 128>>(points_h, &centroids, k, &clusterChanged); //why + 127?
	    cudaDeviceSynchronize();


	    CUDA_CHECK_RETURN(cudaMemcpy(points_h, points_d, sizeof(Point) * DATA_SIZE, cudaMemcpyDeviceToHost)); //device to host
	    CUDA_CHECK_RETURN(cudaMemcpy(centroids_h, centroids_d, sizeof(Point) * k, cudaMemcpyHostToDevice));


	    if (!clustersChanged) {
	        break;          // exit if clusters has not been changed
	    }

	    //Step 3: update centroids
	    updateCentroids(points_d, &centroids_d, k);

	    lastEpoch = ep;
	 }

	CUDA_CHECK_RETURN(cudaMemcpy(points_h, points_d, sizeof(Point) * DATA_SIZE, cudaMemcpyDeviceToHost));
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
