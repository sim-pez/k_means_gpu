#include <vector>
#include "Point.h"

using namespace std;

float* readCsv();

void writeCsv(float* points, float* centroids, int iteration, int k);

void initialize();
