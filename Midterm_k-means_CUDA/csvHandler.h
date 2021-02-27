#include <vector>
#include "Point.h"

using namespace std;

vector<Point> readCsv();

void writeCsv(vector<Point>* points, vector<Point>* centroids, int iteration, int k);

void initialize();
