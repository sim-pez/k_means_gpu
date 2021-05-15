#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

#include "csvHandler.h"

using namespace std;

void readCsv(float *x, float *y, float *z, int n) {
    string line;
    ifstream file("input/dataset.csv", ifstream::in);

    for (int i = 0; i < n; i++){
    	getline(file, line);
        stringstream lineStream(line);
        string bit;
        getline(lineStream, bit, ',');
        x[i] = stof(bit);
        getline(lineStream, bit, ',');
        y[i] = stof(bit);
        getline(lineStream, bit, ',');
        z[i] = stof(bit);
    }
    file.close();
}

void writeCsv(float* pointsX, float* pointsY, float* pointsZ, float* centroidsX, float* centroidsY, float* centroidsZ, int* clusters, int n, int k) {
    ofstream fileIterations("points.csv", ifstream::out);
    for (int i = 0; i < n; i++ ){
    	fileIterations << pointsX[i] << "," << pointsY[i] << "," << pointsZ[i ] << "," << clusters[i] << "\n";
    }
    fileIterations.close();

    ofstream fileCentroids("centroids.csv", ifstream::out);
    for (int i = 0; i < k; i++ ){
        fileCentroids << centroidsX[i] << "," << centroidsY[i] << "," << centroidsZ[i] << "\n";
    }
    fileIterations.close();
}

void writeDurationCsv(int* meanVectorDuration) {
	ofstream fileDuration("durationCUDA.csv", ifstream::out);
	for (int i=0; i<10; i++) {
		fileDuration << meanVectorDuration[i] << "\n";
	}
	fileDuration.close();
}
