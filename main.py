import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
# File reading and assigning the values
start = time.time()
with open(input("Enter a file name: "), 'r') as file:
    line = file.readline()
    listPoints = []
    x_points = []
    y_points = []
    while file and line:
        line = line.strip()
        line_sep = line.split(" ")
        digits = [int(x) for x in line_sep if x.isdigit()]
        listPoints.append((digits[1], digits[2]))
        x_points.append(digits[1])
        y_points.append(digits[2])
        line = file.readline()

X = np.array(listPoints)
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0, n_init='auto')
y = kmeans.fit_predict(X)
cent = kmeans.cluster_centers_
plt.scatter(X[y==0, 0], X[y==0, 1], s=1, c='red', label ='Cluster 1')
plt.scatter(X[y==1, 0], X[y==1, 1], s=1, c='blue', label ='Cluster 2')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=20, c='yellow', label = 'Centroids')
plt.show()
end = time.time()
elapsed_time = end - start
print('Execution time:', elapsed_time, 'seconds')