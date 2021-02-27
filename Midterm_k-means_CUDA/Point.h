#ifndef KMEANS_POINT_H
#define KMEANS_POINT_H


class Point {
public:
    Point(float x, float y, float z);

    float getX() const;

    float getY() const;

    float getZ() const;

    void setX(float xVal);

    void setY(float yVal);

    void setZ(float zVal);

    int getCluster() const;

    void setCluster(int c);

    int getOldCluster() const;

    void setOldCluster(int c);

    float getClusterDistance() const;

    void setClusterDistance(float minDistance);

private:
    float x, y, z;
    int cluster;
    int oldcluster;
    float clusterDistance;
};


#endif //KMEANS_POINT_H
