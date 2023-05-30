import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
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
kmeans = KMeans(n_clusters=8, init='k-means++', random_state=0, n_init='auto')
y = kmeans.fit_predict(X)
cent = kmeans.cluster_centers_
plt.scatter(X[y==0, 0], X[y==0, 1], s=1, c='red', label ='Cluster 1')
plt.scatter(X[y==1, 0], X[y==1, 1], s=1, c='blue', label ='Cluster 2')
plt.scatter(X[y==2, 0], X[y==2, 1], s=1, c='green', label ='Cluster 3')
plt.scatter(X[y==3, 0], X[y==3, 1], s=1, c='purple', label ='Cluster 4')
plt.scatter(X[y==4, 0], X[y==4, 1], s=1, c='pink', label ='Cluster 5')
plt.scatter(X[y==5, 0], X[y==5, 1], s=1, c='black', label ='Cluster 6')
plt.scatter(X[y==6, 0], X[y==6, 1], s=1, c='cyan', label ='Cluster 7')
plt.scatter(X[y==7, 0], X[y==7, 1], s=1, c='magenta', label ='Cluster 8')

# optics = OPTICS(min_samples=1000, xi=0.05, min_cluster_size=0.05).fit(X)
# labels = optics.labels_
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
# plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)

plt.show()
end = time.time()
elapsed_time = end - start
print('Execution time:', elapsed_time, 'seconds')
