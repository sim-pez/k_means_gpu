#ifndef KMEANS_DISTANCE_H
#define KMEANS_DISTANCE_H

#endif //KMEANS_DISTANCE_H

#include "Point.h"
#include <vector>

float distance3d(float x1, float x2, float x3, float y1, float y2, float y3);

float distance3d(Point p1, Point p2);

float mean(std::vector<float>* v);
