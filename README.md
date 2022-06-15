# Intro

This is an implementation of [k-means  clustering algorithm](https://en.wikipedia.org/wiki/K-means_clustering) using GPU acceleration with CUDA C++.

It is made for execution time comparison wrt CPU-only [sequential version](https://github.com/MarcoSolarino/Midterm_Parallel_Computing_K-means). For this reason the algorithm will not end when reaching convergence, but you need to specify the number of iterations when launching the program.

# Generating dataset

You can generate an _N_ points dataset using ```datasetgen.py```. You have to write also the number of clusters _K_ and the standard deviation. The command will be like:
```
python datasetgen.py N K STD
```
example:
```
python datasetgen.py 1000 3 0.45
```

### Note
The code is made to work on 3 axis but the script will generate points with the third coordinate equal to _0.0_. This is to ease the result checking with ```plot.py```

# Usage 

```
./Kmeans N K I
```
Where _N_ is the number of points to read from dataset, _K_ is the number of clusters and _I_ is the number of iterations


# Plotting
You can check output results with
```
python plot.py
```

# Other versions
- CPU-only [sequential version](https://github.com/MarcoSolarino/Midterm_Parallel_Computing_K-means)
- there is also a [distributed system version](https://github.com/sim-pez/k_means_distributed)

# Acknowledgments
Parallel Computing - Computer Engineering Master Degree @[University of Florence](https://www.unifi.it/changelang-eng.html).
