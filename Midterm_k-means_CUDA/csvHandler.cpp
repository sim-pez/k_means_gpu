#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <filesystem>

#include "csvHandler.h"
#include "definitions.h"

float *readCsv() {
    vector<Point> points;
    string line;
    ifstream file("../dataset.csv", ifstream::in);

    float *data = (float *) malloc(sizeof(float) * DATA_SIZE);

    for (int i = 0; i < DATA_SIZE; i = i + 3 ){
        getline(file, line);
        stringstream lineStream(line);
        string bit;
        getline(lineStream, bit, ','); // x
        data[i] = stof(bit);
        getline(lineStream, bit, ','); // y
        data[i + 1] = stof(bit);
        getline(lineStream, bit, ','); // z
        data[i + 2] = stof(bit);
    }
    file.close();
    return data;
}

void writeCsv(vector<Point>* points, vector<Point>* centroids, int iteration, int k) {
    ofstream fileIterations("../output/k" + to_string(k) + "iteration" + to_string(iteration) + ".csv", ifstream::out);
    for(auto &point : *points) {
        fileIterations << point.getX() << "," << point.getY() << "," << point.getZ() << "," << point.getCluster() << "\n";
    }
    fileIterations.close();

    ofstream fileCentroids("../output/k" + to_string(k) + "centroids" + to_string(iteration) + ".csv", ifstream::out);
    for(auto &centroid: *centroids) {
        fileCentroids << centroid.getX() << "," << centroid.getY() << "," << centroid.getZ() << "\n";
    }
    fileIterations.close();

}

void initialize(){
    std::filesystem::remove_all("../output/");
    std::filesystem::create_directory("../output/");
}
