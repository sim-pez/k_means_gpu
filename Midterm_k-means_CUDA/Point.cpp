#include "Point.h"

Point::Point(float x, float y, float z) : x(x), y(y), z(z) {
    cluster = -1;
    oldcluster = -1;
    clusterDistance = __DBL_MAX__;
}

int Point::getCluster() const {
    return cluster;
}

void Point::setCluster(int c) {
    Point::cluster = c;
}

int Point::getOldCluster() const{
    return oldcluster;
}

void Point::setOldCluster(int c) {
    Point::oldcluster = c;
}

float Point::getClusterDistance() const {
    return clusterDistance;
}

void Point::setClusterDistance(float distance) {
    Point::clusterDistance = distance;
}

float Point::getX() const {
    return x;
}

float Point::getY() const {
    return y;
}

float Point::getZ() const {
    return z;
}

void Point::setX(float xVal) {
    Point::x = xVal;
}

void Point::setY(float yVal) {
    Point::y = yVal;
}

void Point::setZ(float zVal) {
    Point::z = zVal;
}
