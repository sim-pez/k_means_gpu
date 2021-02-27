#include "operations.h"
#include <cmath>
#include <vector>

using namespace std;

float distance3d(float x1, float x2, float x3, float y1, float y2, float y3) {
    // compute euclidean distance
    float distance = sqrt(pow(x1 - y1, 2) + pow(x2 - y2, 2) + pow(x3 - y3, 2));
    return distance;
}

float distance3d(Point p1, Point p2){
    float x1 = p1.getX();
    float x2 = p1.getY();
    float x3 = p1.getZ();
    float y1 = p2.getX();
    float y2 = p2.getY();
    float y3 = p2.getZ();
    return distance3d(x1, x2, x3, y1, y2, y3);
}

float mean(vector<float> *v){
    float mean = 0;
    for (auto &value : *v){
        mean += value;
    }
    mean = mean/v->size();
    return mean;
}
