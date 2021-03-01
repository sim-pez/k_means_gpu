#include "operations.h"
#include <cmath>
#include <vector>

using namespace std;

float distance3d(float x1, float x2, float x3, float y1, float y2, float y3) {
    // compute euclidean distance
    float distance = sqrt(pow(x1 - y1, 2) + pow(x2 - y2, 2) + pow(x3 - y3, 2));
    return distance;
}


float mean(vector<float> *v){
    float mean = 0;
    for (auto &value : *v){
        mean += value;
    }
    mean = mean/v->size();
    return mean;
}
