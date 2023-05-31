import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
import math
import gc
from scipy.stats import pearsonr
import scipy.stats as stats
from itertools import cycle
from sklearn.cluster import OPTICS

def calculate_density(elm,cent_x, cent_y):
    density = 0
    for i in range(0, elm.size, 2):
        density += math.sqrt((cent_x - elm[i])**2 + (cent_y - elm[i+1])**2)
    return density/(elm.size/2)

# File reading and assigning the values
colors = ['aqua', 'aquamarine', 'azure', 'beige', 'black', 'blue', 'brown', 'chartreuse', 'chocolate', 'coral', 'crimson', 'cyan', 'darkblue', 'darkgreen', 'fuchsia', 'gold', 'goldenrod', 'green', 'grey', 'indigo', 'ivory', 'khaki', 'lavender', 'lightblue', 'lightgreen', 'lime', 'magenta', 'maroon', 'navy', 'olive', 'orange', 'orangered', 'orchid', 'pink', 'plum', 'purple', 'red','salmon', 'sienna', 'silver', 'tan', 'teal', 'tomato', 'turquoise', 'violet', 'wheat', 'yellow' , 'yellowgreen'  ]
random.shuffle(colors)
start = time.time()
with open(input("Enter a file name: "), 'r') as file:
    line = file.readline()
    listPoints = []
    while file and line:
        line = line.strip()
        line_sep = line.split(" ")
        digits = [int(x) for x in line_sep if x.isdigit()]
        listPoints.append((digits[1], digits[2]))
        line = file.readline()

X = np.array(listPoints, dtype='int')
del listPoints
gc.collect()
cluster_size = math.ceil(X.size/1500)
kmeans = KMeans(n_clusters=cluster_size, init='k-means++', random_state=0, n_init='auto')
y = kmeans.fit_predict(X)
cent = kmeans.cluster_centers_

# std - z score
st_dev = []
pearson_corr = []
norms = []
elements = np.array(X.flatten())
for i in range(cluster_size):
    r = np.round(np.random.rand(), 1)
    g = np.round(np.random.rand(), 1)
    b = np.round(np.random.rand(), 1)
    st_dev.append(calculate_density(elements, cent[i][0], cent[i][1]))
    plt.scatter(X[y == i, 0], X[y == i, 1], s=1, color=[r,g,b])
    print(st_dev[i])
    print(cent[i])
    # pearson_corr.append(pearsonr(X[y==i, 0],X[y==i, 1]))
    # norms.append(stats.norm.pdf(X))
    # scipy.stats.multivariate_normal()

plt.scatter(cent[:, 0], cent[:, 1], s=20, c = 'yellow')

# optics = OPTICS(min_samples=1000, xi=0.05, min_cluster_size=0.05).fit(X)
# labels = optics.labels_
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
# plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)

plt.show()
end = time.time()
elapsed_time = end - start
print('Execution time:', elapsed_time, 'seconds')
