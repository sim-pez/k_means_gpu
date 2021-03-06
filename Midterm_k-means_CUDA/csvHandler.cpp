#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

#include "csvHandler.h"
#include "definitions.h"

float *readCsv() {
    string line;
    ifstream file("dataset.csv", ifstream::in);

    float *data = (float *) malloc(sizeof(float) * DATA_SIZE * 3);

    int i = 0;
    for (int i = 0; i < DATA_SIZE; i++){
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

void writeCsv(float* points, float* centroids, int* clusters, int iteration) {
    ofstream fileIterations("/output/k" + to_string(CLUSTER_NUM) + "iteration" + to_string(iteration) + ".csv", ifstream::out);
    for (int i = 0; i < DATA_SIZE; i++ ){
    	fileIterations << points[i * 3] << "," << points[i * 3 + 1] << "," << points[i * 3 + 2] << "," << clusters[i] << "\n";
    }
    fileIterations.close();

    ofstream fileCentroids("/output/k" + to_string(CLUSTER_NUM) + "centroids" + to_string(iteration) + ".csv", ifstream::out);
    for (int i = 0; i < CLUSTER_NUM; i++ ){
        fileCentroids << centroids[i * 3] << "," << centroids[i * 3 + 1] << "," << centroids[i * 3 + 2] << "\n";
    }
    fileIterations.close();
}

void initialize(){
    //std::filesystem::remove_all("../output/");
    //std::filesystem::create_directory("../output/");
}
