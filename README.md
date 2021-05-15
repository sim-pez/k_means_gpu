This implementation of k-means algorithm is intended for execution time comparison wrt [Sequential version](https://github.com/MarcoSolarino/Midterm_Parallel_Computing_K-means), so you need to specify manually how many iterations you want to execute.

## 1 - Generating dataset
first you need to generate dataset. Change directory to Midterm_k-means_CUDA and then you can use python script
```
python datasetgen.py N K STD
```
where N is the number of points you want to generate, K is the number of clusters and STD is standard deviation of points from clusters. Here is an example:
```
python datasetgen.py 1000 3 0.45
```
## 2 - Run
```
Midterm_k-means_CUDA N K I
```
Where N is the number of points to read from dataset, K is the number of clusters and I the number of iterations. (same as C++ version)

Example:
```
./Kmeans 1000 5 50
```
Will look for 5 clusters through first 1000 points of dataset and it will iterate 50 times. 

## 3 - Check output
After running program you can check output results with
```
python plot.py
```

### Note
The code is made to work with 3D points but datasetgen.py will generate points with z = 0.0. This is intended to ease result checking with plot.py

## Other k-means versions
- [Sequential](https://github.com/MarcoSolarino/Midterm_Parallel_Computing_K-means)
- [Hadoop](https://github.com/daikon899/Midterm_K-means_hadoop)
