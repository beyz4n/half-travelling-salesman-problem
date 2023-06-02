import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
import math
import gc
import random


def calculate_density(elm, cent_x, cent_y):
    density = 0
    for j in range(0, elm.size, 2):
        density += math.sqrt((cent_x - elm[j])**2 + (cent_y - elm[j+1])**2)
    return density/(elm.size/2)


def print_length(cities):
    dist = 0
    for m in range(len(cities)):
        dist += distance_btw_two_cities(m, m - 1, cities)
    print(dist)


# method to swap pairs of nodes according to their index size
def rearrange_nodes(node1, node2, node3, node4, node5, node6):
    nodes = [node1, node2, node3, node4, node5, node6]
    nodes.sort()
    if nodes[0] == 0:
        return nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[0]
    else:
        return nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5]


# method to shift the nodes backward
def shift_nodes(length, node1, node2):
    node1 = node1 - 1
    if node2 == 0:
        node2 = length - 1
    else:
        node2 = node2 - 1
    return node1, node2


# method to calculate Euclidean distance between 2 nodes
def distance_btw_two_cities(city1, city2, cities):
    difference = math.sqrt(
        (x_points[cities[city1]] - x_points[cities[city2]]) ** 2 + (y_points[cities[city1]] - y_points[cities[city2]]) ** 2)
    return difference


# method to calculate Euclidean distance
def distance(city1, city2, city3, city4, city5, city6, cities):
    distance = distance_btw_two_cities(city1, city2, cities) + distance_btw_two_cities(city3, city4, cities) + \
               distance_btw_two_cities(city5, city6, cities)
    return distance


def three_opt(cities):
    length = len(cities)
    # number of iteration for loop
    if length < 3000:
        size = 3000
    else:
        size = len(cities)

    for i in range(0, size):
        # chose 3 node pairs
        # chose first pair
        node_a1 = random.randint(1, length - 1)
        node_a2 = node_a1 + 1
        if node_a2 == length:
            node_a2 = 0  # if it is len which mean is circle completed, actually it is 0th one

        # chose second pair
        node_b1 = random.randint(2, length - 1)
        while node_b1 == node_a1 or node_b1 == node_a2:
            node_b1 = random.randint(2, length - 1)

        node_b2 = node_b1 + 1
        if node_b2 == length:  # if it is len which mean is circle completed, actually it is 0th one
            node_b2 = 0
        if node_b2 == node_a1:  # if there is collision shift back node_b1 and node_b2
            node_b1, node_b2 = shift_nodes(length, node_b1, node_b2)

        # chose third pair
        node_c1 = random.randint(2, length - 1)
        while (node_c1 == node_a1 or node_c1 == node_a2 or node_c1 == node_b1 or node_c1 == node_b2 or
               (node_c1 + 1 == node_b1 and node_c1 - 1 == node_a2) or (node_c1 + 1 == node_a1 and node_c1 - 1 == node_b2)):
            node_c1 = random.randint(2, length - 1)

        node_c2 = node_c1 + 1
        if node_c2 == length:  # if it is len which mean is circle completed, actually it is 0th one
            node_c2 = 0
        if node_c2 == node_b1 or node_c2 == node_a1:  # if there is collision shift back node_c1 and node_c2
            node_c1, node_c2 = shift_nodes(length, node_c1, node_c2)

        # swap pairs of nodes according to their index size
        node_a1, node_a2, node_b1, node_b2, node_c1, node_c2 = rearrange_nodes(node_a1, node_a2, node_b1, node_b2, node_c1, node_c2)

        # current length for 3 path
        current = distance(node_a1, node_a2, node_b1, node_b2, node_c1, node_c2, cities)
        # for option 1
        length_1 = distance(node_a1, node_c1, node_c2, node_a2, node_b1, node_b2, cities)
        # for option 2
        length_2 = distance(node_a1, node_a2, node_c2, node_b2, node_c1, node_b1, cities)
        # for option 3
        length_3 = distance(node_c1, node_c2, node_a1, node_b1, node_a2, node_b2, cities)
        # for option 4
        length_4 = distance(node_a2, node_c1, node_a1, node_b1, node_c2, node_b2, cities)
        # for option 5
        length_5 = distance(node_a2, node_b2, node_a1, node_c1, node_b1, node_c2, cities)
        # for option 6
        length_6 = distance(node_a2, node_c2, node_c1, node_b1, node_a1, node_b2, cities)
        # for option 7
        length_7 = distance(node_a2, node_c1, node_b1, node_c2, node_a1, node_b2, cities)

        # find min length
        min_length = min(length_1, length_2, length_3, length_4, length_5, length_6, length_7, current)

        # create a tour of less length
        if min_length == length_1:
            if node_c2 == 0:
                cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_a2: node_c1 + 1])))
            else:
                cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_a2: node_c1 + 1]), cities[node_c2:]))
        elif min_length == length_2:
            if node_c2 == 0:
                cities = np.concatenate((cities[: node_b1 + 1], np.flip(cities[node_b2: node_c1 + 1])))
            else:
                cities = np.concatenate((cities[: node_b1 + 1], np.flip(cities[node_b2: node_c1 + 1]), cities[node_c2:]))
        elif min_length == length_3:
            cities = np.concatenate((cities[:node_a1 + 1], np.flip(cities[node_a2: node_b1 + 1]), cities[node_b2:]))
        elif min_length == length_4:
            if node_c2 == 0:
                cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_a2: node_b1 + 1]),
                                         np.flip(cities[node_b2: node_c1 + 1])))
            else:
                cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_a2: node_b1 + 1]),
                                         np.flip(cities[node_b2: node_c1 + 1]), cities[node_c2:]))
        elif min_length == length_5:
            if node_c2 == 0:
                cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_b2: node_c1 + 1]),
                                         cities[node_a2:node_b1 + 1]))
            else:
                cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_b2: node_c1 + 1]),
                                         cities[node_a2:node_b1 + 1], cities[node_c2:]))
        elif min_length == length_6:
            if node_c2 == 0:
                cities = np.concatenate((cities[: node_a1 + 1], cities[node_b2: node_c1 + 1],
                                         np.flip(cities[node_a2: node_b1 + 1])))
            else:
                cities = np.concatenate((cities[: node_a1 + 1], cities[node_b2: node_c1 + 1],
                                         np.flip(cities[node_a2: node_b1 + 1]), cities[node_c2:]))
        elif min_length == length_7:
            if node_c2 == 0:
                cities = np.concatenate((cities[: node_a1 + 1], cities[node_b2: node_c1 + 1],
                                         cities[node_a2: node_b1 + 1]))
            else:
                cities = np.concatenate((cities[: node_a1 + 1], cities[node_b2: node_c1 + 1],
                                         cities[node_a2: node_b1 + 1], cities[node_c2:]))
    return cities


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

size = int(digits[0]) + 1
X = np.array(listPoints, dtype='int')
del listPoints
gc.collect()
cluster_size = math.ceil(size/750)
kmeans = KMeans(n_clusters=cluster_size, init='k-means++', random_state=0, n_init='auto')
y = kmeans.fit_predict(X)
cent = kmeans.cluster_centers_

# std - z score
if cluster_size > 1:
    densities = []

    for pt in range(cluster_size):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        elements = np.array(X[y == pt].flatten())
        densities.append(calculate_density(elements, cent[pt][0], cent[pt][1]))
        plt.scatter(X[y == pt, 0], X[y == pt, 1], s=1, color=[r, g, b])
        # pearson_corr.append(pearsonr(X[y==i, 0],X[y==i, 1]))
        # norms.append(stats.norm.pdf(X))
        # scipy.stats.multivariate_normal()
    density_arr = np.array(densities)
    ids = np.arange(cluster_size)
    features = np.column_stack((ids, density_arr))
    clusters = np.hstack((cent, features))
    print(clusters)
else:
    plt.scatter(cent[:, 0], cent[:, 1], s=20, c='yellow')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=1, c='crimson')

# optics = OPTICS(min_samples=1000, xi=0.05, min_cluster_size=0.05).fit(X)
# labels = optics.labels_
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
# plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)

plt.show()
end = time.time()
elapsed_time = end - start
print('Execution time:', elapsed_time, 'seconds')
