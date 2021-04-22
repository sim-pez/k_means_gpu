import sys
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as sk
import pandas as pd
import os

points_num = int(sys.argv[1])
k = int(sys.argv[2])
std = float(sys.argv[3])

X, y = sk.make_blobs(n_samples=points_num, n_features=2, centers=k, cluster_std=std)
array = X.tolist()
for a in array:
    a.append(0)
nparray = np.array(array)

# df = pd.DataFrame(nparray)
# lenght = len(df)
	
# for i in range(lenght):
#     plt.scatter(df.iloc[i, 0], df.iloc[i, 1])
# plt.grid()
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()	

if not os.path.exists('input'):
    os.makedirs('input')
np.savetxt("input/dataset.csv", nparray, delimiter=",", fmt='%f')
print('dataset of ' + str(points_num) + ' points created!')