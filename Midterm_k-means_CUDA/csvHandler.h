#include <vector>
#include "Point.h"

using namespace std;

float* readCsv();

void writeCsv(vector<Point>* points, vector<Point>* centroids, int iteration, int k);

void initialize();
