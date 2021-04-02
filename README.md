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
...in progress

## 3 - Plotting
```
python plot.py
```
After running program you can plot result

## Other k-means versions
- [Sequential](https://github.com/MarcoSolarino/Midterm_Parallel_Computing_K-means)
- [Hadoop](https://github.com/daikon899/Midterm_K-means_hadoop)
