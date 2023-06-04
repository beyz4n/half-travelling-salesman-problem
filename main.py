import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
import math
import gc
import random
import networkx as nx


def calculate_density(elm, cent_x, cent_y):
    density = 0
    for j in range(0, elm.size, 2):
        density += math.sqrt((cent_x - elm[j])**2 + (cent_y - elm[j+1])**2)
    return density/(elm.size/2)


def sort_list(elm):
    list = elm
    temp = 0
    j = 0
    index = 0

    for i in range( int( list.size/4) ):
        j = i
        temp = list[i,1]
        index = i
        while(j+1< int( list.size/4) ):
            if(list[j+1,1]<temp):
                temp = list[j+1,1]
                index = j+1
            j += 1

        temp = list[i,1]
        list[i,1] = list[index,1]
        list[index, 1] = temp
        temp = list[i,0]
        list[i,0] = list[index,0]
        list[index, 0] = temp
        temp = list[i,2]
        list[i,2] = list[index,2]
        list[index, 2] = temp
        temp = list[i,3]
        list[i,3] = list[index,3]
        list[index, 3] = temp

    i = 1
    counter = 0
    while(i< int(list.size/4)):
        if(list[i-1,0] == list[i,0]):
            j = i
            while(j < int(list.size/4) and list[j-1,0] == list[j,0] ):
                counter += 1
                j += 1
            j = i
            while(j< i+counter):
                k = j
                temp = list[j,0]
                while(k< i+counter):
                    if(list[k,0] < temp):
                        temp = list[k,0]
                        index = k
                    k += 1
                temp = list[j,0]
                list[j,0] = list[index,0]
                list[index,0] = temp
                temp = list[j,1]
                list[j,1] = list[index,1]
                list[index,1] = temp
                temp = list[j,2]
                list[j,2] = list[index,2]
                list[index,2] = temp
                temp = list[j,3]
                list[j,3] = list[index,3]
                list[index,3] = temp
                j += 1
        i += 1

    print()
    print(list)
    return list


temp_list = [] # bunu temp olarak kullan
def find_best(elm, index, new_best_list, prev_index):
    list = elm
    i = index
    while(i< int(list.size/4) ):
        if( i + (int(list.size/8)-index) < int(list.size/4) and index < i and prev_index < i ):
            temp_list[index-1, 0] = list[i,0]
            temp_list[index-1, 1] = list[i,1]
            temp_list[index-1, 2] = list[i,2]
            temp_list[index-1, 3] = list[i,3]
        if(index<int(list.size/8) and  i + (int(list.size/8)-index) < int(list.size/4) and prev_index < i ):
            index += 1
            prev_index = i
            new_best_list = find_best(list, index, new_best_list, prev_index)
            index -= 1
            if(index == 0):
                return new_best_list

        if(index == int(list.size/8) and prev_index < i ):
            if( get_weight(temp_list) < get_weight(new_best_list) ):
                new_best_list = temp_list
        i += 1


    return new_best_list

def get_weight(list):
    weight = 0
    temp_density = list[0,3]
    multiplier = 0
    i = 1
    while(i < int(list.size/4) -1 ):
        multiplier = list[i,3] - temp_density
        temp_density = list[i,3]
        weight += math.sqrt( (list[i,0]-list[i-1,0])**2 +  (list[i,1]-list[i-1,1])**2 ) * abs(multiplier)
        i += 1
    multiplier = list[0,3] - list[i,3]
    weight += math.sqrt( (list[0,0]-list[i,0])**2 +  (list[0,1]-list[i,1])**2 ) * abs(multiplier)
    return weight

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


def calculate_distance(x1, y1, x2, y2):
    return (x2 - x1) ** 2 + (y2 - y1) ** 2


def create_weighted_graph(x_coords, y_coords):
    num_cities = len(x_coords)
    graph = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                graph[i, j] = calculate_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j])
            else:
                graph[i, j] = float('inf')

    return graph


def prims_algorithm(graph):
    num_vertices = graph.shape[0]
    # Create an array to keep track of visited vertices
    visited = np.zeros(num_vertices, dtype=bool)
    # Create an array to store the parent of each vertex in the MST
    parent = np.zeros(num_vertices, dtype=int)
    # Create an array to store the minimum weight for each vertex
    weight = np.full(num_vertices, float('inf'))

    # Start with the first vertex
    weight[0] = 0
    parent[0] = -1

    for k in range(num_vertices):
        # Find the vertex with the minimum weight that has not been visited
        min_weight = float('inf')
        min_vertex = -1
        for v in range(num_vertices):
            if not visited[v] and weight[v] < min_weight:
                min_weight = weight[v]
                min_vertex = v

        visited[min_vertex] = True

        # Update the weight and parent for neighboring vertices
        for v in range(num_vertices):
            if (
                    not visited[v] and
                    graph[min_vertex][v] != 0 and
                    graph[min_vertex][v] < weight[v]
            ):
                weight[v] = graph[min_vertex][v]
                parent[v] = min_vertex

    return parent


def plot_mst(x_coords, y_coords, parent, odd, matching):
    plt.figure()
    for i in range(1, len(parent)):
        plt.plot([x_coords[i], x_coords[parent[i]]], [y_coords[i], y_coords[parent[i]]], 'r-')

    for i in range(len(matching)):
        plt.plot([x_coords[matching[i][0]], x_coords[matching[i][1]]],
                 [y_coords[matching[i][0]], y_coords[matching[i][1]]], 'r-', color='purple')

    plt.scatter(x_coords, y_coords, color='b')

    oddCoordsX = []
    oddCoordsY = []
    for i in range(len(odd)):
        oddCoordsX.append(x_coords[odd[i]])
        oddCoordsY.append(y_coords[odd[i]])

    plt.scatter(oddCoordsX, oddCoordsY, color='yellow')
    plt.show()


def plot_Multi(x_coords, y_coords, parent):
    plt.figure()
    for i in range(1, len(parent)):
        for j in range(1, len(parent[i])):
            plt.plot([x_coords[i], x_coords[parent[i][j]]], [y_coords[i], y_coords[parent[i][j]]], 'r-')

    plt.scatter(x_coords, y_coords, color='b')

    plt.show()


def plot_Euler(x_coords, y_coords, parent):
    plt.figure()
    for i in range(len(parent)):
        plt.plot([x_coords[parent[i - 1]], x_coords[parent[i]]], [y_coords[parent[i - 1]], y_coords[parent[i]]], 'r-',
                 color=random.choice(['red', 'green', 'blue', 'yellow', 'black', 'purple', 'pink', 'orange']))

    plt.scatter(x_coords, y_coords, color='b')

    plt.show()


nx.graph


def find_perfect_matching(odd_vertices, matrix):
    edges = []
    indexes = []
    for i in range(len(odd_vertices)):
        if i in indexes:
            continue
        min = float('inf')
        min_index = -1
        for j in range(len(odd_vertices)):
            if i != j and j not in indexes and matrix[odd_vertices[i]][odd_vertices[j]] < min:
                min = matrix[odd_vertices[i]][odd_vertices[j]]
                min_index = j
        indexes.append(min_index)
        indexes.append(i)
        edges.append((odd_vertices[i], odd_vertices[min_index]))
    return edges


def find_odd_degree_nodes(parent_data):
    # Initialize a dictionary to store the in-degree and out-degree of each vertex
    in_degree = {}
    out_degree = {}

    # Count the in-degree and out-degree of each vertex
    for child, parent in enumerate(parent_data):
        if parent != -1:
            in_degree[child] = in_degree.get(child, 0) + 1
            out_degree[parent] = out_degree.get(parent, 0) + 1

    # Handle the first element separately
    first_element = parent_data[0]
    if first_element == -1:
        in_degree[0] = 0

    # Find the odd degree nodes
    odd_degree_nodes = [node for node in in_degree.keys() if (in_degree[node] + out_degree.get(node, 0)) % 2 != 0]

    return odd_degree_nodes


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
        plt.scatter(cent[:, 0], cent[:, 1], s=20, c='yellow')
        # pearson_corr.append(pearsonr(X[y==i, 0],X[y==i, 1]))
        # norms.append(stats.norm.pdf(X))
        # scipy.stats.multivariate_normal()
    density_arr = np.array(densities)
    ids = np.arange(cluster_size)
    features = np.column_stack((ids, density_arr))
    clusters = np.hstack((cent, features))
    print(clusters)
    sort_list(clusters)
    best_list = []
    for i in range(int(cluster_size / 2)):
        best_list = np.append(best_list, np.copy(clusters[i]), axis=0)
        temp_list = np.append(temp_list, np.copy(clusters[i]), axis=0)

    best_list = best_list.reshape(-1, 4)
    temp_list = temp_list.reshape(-1, 4)
    best_list = find_best(clusters, 0, best_list, 0)
    print()
    print(best_list)
else:
    # from one cluster choose n/2 nodes
    plt.scatter(cent[:, 0], cent[:, 1], s=20, c='yellow')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=1, c='crimson')
    node_number = len(x_points)
    lookup = []
    cent_x = cent[0][0]
    cent_y = cent[0][1]
    for pts in range(node_number):
        distance_btw_cent_and_node = (x_points[pts] - cent_x)**2 + (y_points[pts] - cent_y)**2
        lookup.append((pts,distance_btw_cent_and_node))
    lookup.sort(key=lambda a: a[1])
    needed_node = math.ceil(node_number/2)
    temp_best_x = []
    temp_best_y = []
    for ptr in range(needed_node):
        temp_best_x.append(x_points[lookup[ptr][0]])
        temp_best_y.append(y_points[lookup[ptr][0]])
    x_points = temp_best_x
    y_points = temp_best_y
    gc.collect()
    plt.scatter(x_points, y_points, s=5, c='blue')





# optics = OPTICS(min_samples=1000, xi=0.05, min_cluster_size=0.05).fit(X)
# labels = optics.labels_
# colors = list(map(lambda x: '#3b4cc0' if x == 1 else '#b40426', labels))
# plt.scatter(X[:,0], X[:,1], c=colors, marker="o", picker=True)
plt.show()
end = time.time()
elapsed_time = end - start
print('Execution time:', elapsed_time, 'seconds')
