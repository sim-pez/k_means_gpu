#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

#include "csvHandler.h"
#include "definitions.h"

float *readCsv(int n) {
	string line;
    ifstream file("input/dataset.csv", ifstream::in);

    float *data = (float *) malloc(sizeof(float) * n * 3);

    int i = 0;
    for (int i = 0; i < n; i++){
    	getline(file, line);
        stringstream lineStream(line);
        string bit;
        getline(lineStream, bit, ','); // x
        data[i * 3] = stof(bit);
        getline(lineStream, bit, ','); // y
        data[i * 3 + 1] = stof(bit);
        getline(lineStream, bit, ','); // z
        data[i * 3 + 2] = stof(bit);
    }
    file.close();
    return data;
}

void writeCsv(float* pointsX, float* pointsY, float* pointsZ, float* centroidsX, float* centroidsY, float* centroidsZ, int* clusters, int n) {
    ofstream fileIterations("output/points.csv", ifstream::out);
    for (int i = 0; i < n; i++ ){
    	fileIterations << pointsX[i] << "," << pointsY[i] << "," << pointsZ[i ] << "," << clusters[i] << "\n";
    }
    fileIterations.close();

    ofstream fileCentroids("output/centroids.csv", ifstream::out);
    for (int i = 0; i < CLUSTER_NUM; i++ ){
        fileCentroids << centroidsX[i] << "," << centroidsY[i] << "," << centroidsZ[i] << "\n";
    }
    fileIterations.close();
}

void writeDurationCsv(int* meanVectorDuration) {
	ofstream fileDuration("durationCUDA.csv", ifstream::out);
	for (int i=0; i<10; i++) { //TODO change 10
		fileDuration << meanVectorDuration[i] << "\n";
	}
	fileDuration.close();
}

void initialize(){
	//std::filesystem::remove_all("../output/");
	//std::filesystem::create_directory("../output/");
}
