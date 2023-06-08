import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
import math
import gc
import random
import networkx as nx


# This function is for calculating the density of clusters using centroid
def calculate_density(elm, cent_x, cent_y):
    # We put the distance from centroid to the points to the density and return it
    density = 0
    for j in range(0, elm.size, 2):
        density += math.sqrt((cent_x - elm[j]) ** 2 + (cent_y - elm[j + 1]) ** 2)
    return density / (elm.size / 2)

# this function is used to sort the cluster array according to their centroids x and y values
def sort_list(elm):
    list = elm
    temp = 0
    j = 0
    index = 0
    # we first sort them according to their y values
    for i in range(int(cluster_size)):
        j = i
        temp = list[i, 1]
        index = i
        while j + 1 < int(cluster_size):
            if list[j + 1, 1] < temp:
                temp = list[j + 1, 1]
                index = j + 1
            j += 1
        
        temp = list[i, 1]
        list[i, 1] = list[index, 1]
        list[index, 1] = temp
        temp = list[i, 0]
        list[i, 0] = list[index, 0]
        list[index, 0] = temp
        temp = list[i, 2]
        list[i, 2] = list[index, 2]
        list[index, 2] = temp
        temp = list[i, 3]
        list[i, 3] = list[index, 3]
        list[index, 3] = temp
        temp = list[i, 4]
        list[i, 4] = list[index, 4]
        list[index, 4] = temp
    # after that we check if it has multiple clusters with the same y value
    # if it does we sort them according to their x values
    i = 1
    counter = 0
    while i < int(cluster_size):
        if list[i - 1, 0] == list[i, 0]:
            j = i
            while j < int(cluster_size) and list[j - 1, 0] == list[j, 0]:
                counter += 1
                j += 1
            j = i
            while j < i + counter:
                k = j
                temp = list[j, 0]
                while k < i + counter:
                    if list[k, 0] < temp:
                        temp = list[k, 0]
                        index = k
                    k += 1
                temp = list[j, 0]
                list[j, 0] = list[index, 0]
                list[index, 0] = temp
                temp = list[j, 1]
                list[j, 1] = list[index, 1]
                list[index, 1] = temp
                temp = list[j, 2]
                list[j, 2] = list[index, 2]
                list[index, 2] = temp
                temp = list[j, 3]
                list[j, 3] = list[index, 3]
                list[index, 3] = temp
                temp = list[j, 4]
                list[j, 4] = list[index, 4]
                list[index, 4] = temp
                j += 1
        i += 1

    return list


temp_list = []  # use to store temp half lists

# this tries all combinations of clusters and returns the best one
def find_best(elm, new_best_list, temp_list):
    cluster_size = math.ceil(elm.size / 5)
    arr = range(cluster_size)
    best_cluster_size = math.ceil(new_best_list.size / 5)
    
    
    
    pool = tuple(range(cluster_size))
    n = len(pool)
    r = best_cluster_size
    # here we calculate all the possible combinations and chose the best one
    if r > n:
        return
    indices = list(range(r))
    begin = list(pool[i] for i in indices)
    # setting the current best weight
    best_weight = get_weight(new_best_list)
    if math.ceil(size / 2) <= get_num_of_city(temp_list):
            if get_weight(temp_list) < best_weight:
                new_best_list = temp_list
                best_weight = get_weight(temp_list)
    temp_list = []
    temp_list = np.array(elm[begin, :])
    # here we try all the combinations
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            gc.collect()
            return new_best_list
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        end = list(pool[i] for i in indices)
        # here we check if the required city size is  matched, if so we consider if it might be better than the list we already have
        if math.ceil(size / 2) <= get_num_of_city(temp_list):
            if get_weight(temp_list) < best_weight:
                # setting the new best list 
                new_best_list = temp_list
                best_weight = get_weight(temp_list)
        temp_list = []
        temp_list = np.array(elm[end, :])
    # backup return
    gc.collect()
    return new_best_list

# this is a simple method to calculate the total number of cities in the given cluster array 
def get_num_of_city(elm):
    num = 0
    i = 0
    elm_size = int(elm.size / 5)
    while i < elm_size:
        num += elm[i, 4]
        i += 1
    return num

# here we calculate the weight for the given array of clusters
def get_weight(list):
    weight = 0
    temp_density = list[0,3]
    temp_density_2 = 0
    i = 1
    list_size = int(list.size/5)
    # since we sorted the cluster array according to the x and y of their centroids, we roughly should have the nearest clusters in series
    # we calculate the weight by multipliying the distance between two centroids and multiplying it with the clusters std
    # so the lower the weight the better the cluster array
    while(i < list_size ):
        temp_density_2 = list[i,3]
        weight += math.sqrt( (list[i,0]-list[i-1,0])**2 +  (list[i,1]-list[i-1,1])**2 ) * temp_density_2 * temp_density 
        temp_density = temp_density_2
        i += 1


    return weight

# this function is used to terminate the clusters to the desired level which is half of the city size
def terminate_clusters(best_cluster, X, y):
    # initilizing a variable to hold the cluster size which will be terminated
    best_cluster_size = int(best_cluster.size/5)
    
    i = 0
    center_X = 0
    center_Y = 0
    city_size = get_num_of_city(best_cluster)
    # here we find the centroid of all cities
    while i < best_cluster_size:
        center_X += best_cluster[i, 0] * best_cluster[i, 4]
        center_Y += best_cluster[i, 1] * best_cluster[i, 4]
        i += 1
    center_X = center_X / city_size
    center_Y = center_Y / city_size

    temp = 0
    j = 0
    index = 0
    # here we sort the clusters according to their standart deviation from biggest to lowest
    for i in range(best_cluster_size):
        j = i
        temp = best_cluster[i, 3]
        index = i
        while j + 1 < best_cluster_size:
            if best_cluster[j + 1, 3] > temp:
                temp = best_cluster[j + 1, 3]
                index = j + 1
            j += 1

        temp = best_cluster[i, 1]
        best_cluster[i, 1] = best_cluster[index, 1] # we have the y variable of the centroid of the cluster in the index 1
        best_cluster[index, 1] = temp
        temp = best_cluster[i, 0]
        best_cluster[i, 0] = best_cluster[index, 0] # we have the x variable of the centroid of the cluster in the index 0
        best_cluster[index, 0] = temp
        temp = best_cluster[i, 2]
        best_cluster[i, 2] = best_cluster[index, 2] # we have the id of the cluster at index 2
        best_cluster[index, 2] = temp
        temp = best_cluster[i, 3]
        best_cluster[i, 3] = best_cluster[index, 3] # we have the standart deviation at index 3
        best_cluster[index, 3] = temp
        temp = best_cluster[i, 4]
        best_cluster[i, 4] = best_cluster[index, 4] # we have the city size at index 4
        best_cluster[index, 4] = temp


    min_city_size = math.ceil(size / 2) # storing the total city size
    # here we do the trimming proccess 
    while(min_city_size < city_size):

        if(min_city_size < city_size - best_cluster[0,4] ): # we check if can delete whole clusters
            # we delete the hole cluster if we can and update the city size and cluster size
            best_cluster = np.delete(best_cluster, 0, 0 )
            city_size = get_num_of_city(best_cluster)
            best_cluster_size = int(best_cluster.size/5)
        else:
            # if we cannot delete the whole cluster we take the required amount of cities from the remaining cluster with the biggest std
            # here we get the x and y values for that cluster
            x_axis = np.copy(X[y == best_cluster[0, 2], 0].flatten())
            y_axis = np.copy(X[y == best_cluster[0, 2], 1].flatten())

            x_axis = x_axis.flatten()
            y_axis = y_axis.flatten()

            x_point_length = len(x_axis)
            # here we sort the x and y values according to their distance to the centroid of all cities
            i = 0
            for i in range(int(x_point_length)):
                j = i
                temp_X = x_axis[0]
                temp_Y = y_axis[0]
                index = i
                while (j + 1 < x_point_length):
                    if ((y_axis[j + 1] - center_Y) ** 2 + (x_axis[j + 1] - center_X) ** 2 < (temp_X - center_X) ** 2 +  (temp_Y - center_Y) ** 2):
                        temp_X = x_axis[j + 1]
                        temp_Y = y_axis[j + 1]
                        index = j + 1
                    j += 1

                temp = x_axis[i]
                x_axis[i] = x_axis[index]
                x_axis[index] = temp
                temp = y_axis[i]
                y_axis[i] = y_axis[index]
                y_axis[index] = temp
            # we calculate how many cities we need from that cluster
            needed_city = min_city_size - (city_size - best_cluster[0,4] )
            i = 0
            temp_X = []
            temp_Y = []
            # we store the needed amount of cities
            while(i<needed_city):
                temp_X.append(x_axis[i])
                temp_Y.append(y_axis[i])
                i += 1
            break

    # here we add the remaining cities from the other clusters remaining in the best cluster array
    i = 0
    x_axis = temp_X
    y_axis = temp_Y
    for i in range(1, best_cluster_size):
        x_axis = np.append(x_axis, X[y == best_cluster[i, 2], 0].flatten())
        y_axis = np.append(y_axis, X[y == best_cluster[i, 2], 1].flatten())


    flat_XY = np.array(x_axis)
    flat_XY = np.append(flat_XY, y_axis)
    return flat_XY # we return the x and y values for the clusters






def get_actual_dist(city1, city2, cities):
    difference = math.sqrt(graph[cities[city1]][cities[city2]])
    return difference




# method to find length of the tsp tour
def get_length(cities):
    dist = 0
    for m in range(len(cities)):
        dist += int(round(get_actual_dist(m, m - 1, cities)))
    return dist


# method to swap pairs of nodes according to their index size for 3 opt
def rearrange_nodes_for_6_nodes(node1, node2, node3, node4, node5, node6):
    nodes = [node1, node2, node3, node4, node5, node6]
    nodes.sort()
    if nodes[0] == 0:
        return nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[0]
    else:
        return nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5]

# method to swap pairs of nodes according to their index size for 2 opt
def rearrange_nodes_for_4_nodes(node1, node2, node3, node4):
    nodes = [node1, node2, node3, node4]
    nodes.sort()
    if nodes[0] == 0:
        return nodes[1], nodes[2], nodes[3], nodes[0]
    else:
        return nodes[0], nodes[1], nodes[2], nodes[3]

# method to shift the nodes backward
def shift_nodes(length, node1, node2):
    node1 = node1 - 1
    if node2 == 0:
        node2 = length - 1
    else:
        node2 = node2 - 1
    return node1, node2
graph = []
# method to calculate Euclidean distance between 2 nodes
def distance_btw_two_cities(city1, city2, cities):
    difference = graph[cities[city1]][cities[city2]]
    # difference = (int(x_points[cities[city1]]) - int(x_points[cities[city2]])) ** 2 \
    #               + (int(y_points[cities[city1]]) - int(y_points[cities[city2]])) ** 2
    return difference

# method to calculate Euclidean distance for 3 opt
def distance_3opt(city1, city2, city3, city4, city5, city6, cities):
    distance = distance_btw_two_cities(city1, city2, cities) + distance_btw_two_cities(city3, city4, cities) + \
               distance_btw_two_cities(city5, city6, cities)
    return distance

# method to calculate Euclidean distance for 2 opt
def distance_2opt(city1, city2, city3, city4, cities):
    distance = distance_btw_two_cities(city1, city2, cities) + distance_btw_two_cities(city3, city4, cities)
    return distance

def three_opt(cities):
    length = len(cities)
    size = length*2000

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
               (node_c1 + 1 == node_b1 and node_c1 - 1 == node_a2) or (
                       node_c1 + 1 == node_a1 and node_c1 - 1 == node_b2)):
            node_c1 = random.randint(2, length - 1)

        node_c2 = node_c1 + 1
        if node_c2 == length:  # if it is len which mean is circle completed, actually it is 0th one
            node_c2 = 0
        if node_c2 == node_b1 or node_c2 == node_a1:  # if there is collision shift back node_c1 and node_c2
            node_c1, node_c2 = shift_nodes(length, node_c1, node_c2)

        # swap pairs of nodes according to their index size
        node_a1, node_a2, node_b1, node_b2, node_c1, node_c2 = rearrange_nodes_for_6_nodes(node_a1, node_a2, node_b1, node_b2,
                                                                               node_c1, node_c2)

        # current length for 3 path
        current = distance_3opt(node_a1, node_a2, node_b1, node_b2, node_c1, node_c2, cities)
        # for option 1
        length_1 = distance_3opt(node_a1, node_c1, node_c2, node_a2, node_b1, node_b2, cities)
        # for option 2
        length_2 = distance_3opt(node_a1, node_a2, node_c2, node_b2, node_c1, node_b1, cities)
        # for option 3
        length_3 = distance_3opt(node_c1, node_c2, node_a1, node_b1, node_a2, node_b2, cities)
        # for option 4
        length_4 = distance_3opt(node_a2, node_c1, node_a1, node_b1, node_c2, node_b2, cities)
        # for option 5
        length_5 = distance_3opt(node_a2, node_b2, node_a1, node_c1, node_b1, node_c2, cities)
        # for option 6
        length_6 = distance_3opt(node_a2, node_c2, node_c1, node_b1, node_a1, node_b2, cities)
        # for option 7
        length_7 = distance_3opt(node_a2, node_c1, node_b1, node_c2, node_a1, node_b2, cities)

        # find min length
        min_length = min(length_1, length_2, length_3, length_4, length_5, length_6, length_7, current)

        # create a tour of less length
        if min_length == length_1:
            if node_c2 == 0:
                cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_a2: node_c1 + 1])))
            else:
                cities = np.concatenate(
                    (cities[: node_a1 + 1], np.flip(cities[node_a2: node_c1 + 1]), cities[node_c2:]))
        elif min_length == length_2:
            if node_c2 == 0:
                cities = np.concatenate((cities[: node_b1 + 1], np.flip(cities[node_b2: node_c1 + 1])))
            else:
                cities = np.concatenate(
                    (cities[: node_b1 + 1], np.flip(cities[node_b2: node_c1 + 1]), cities[node_c2:]))
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


def two_opt(cities):
    length = len(cities)
    for i in range(1, length):
        for j in range(1, length):
            # chose 2 node pairs
            # chose first pair
            node_a1 = i
            node_a2 = node_a1 + 1
            if node_a2 == length:
                node_a2 = 0  # if it is len which mean is circle completed, actually it is 0th one

            # chose second pair
            node_b1 = j
            node_b2 = node_b1 + 1
            if node_b2 == length:  # if it is len which mean is circle completed, actually it is 0th one
                node_b2 = 0

            # if there is a collision
            if node_b1 == node_a1 or node_b1 == node_a2 or node_b2 == node_a1 or node_b2 == node_a2:
                continue

            # swap pairs of nodes according to their index size
            node_a1, node_a2, node_b1, node_b2 = rearrange_nodes_for_4_nodes(node_a1, node_a2, node_b1, node_b2)

            # current length for 2 path
            current = distance_2opt(node_a1, node_a2, node_b1, node_b2, cities)
            # new path length for 2 path
            new = distance_2opt(node_a1, node_b1, node_a2, node_b2, cities)

            if new < current: # if the newly found path is shorter, update list of cities to visit
                if node_b2 == 0:
                    cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_a2: node_b1 + 1])))
                else:
                    cities = np.concatenate((cities[: node_a1 + 1], np.flip(cities[node_a2: node_b1 + 1]),
                                             cities[node_b2:]))
    return cities

#calculate the distance between two nodes without taking its square root for efficiency
def calculate_distance(x1, y1, x2, y2):
    return (int(x2) - int(x1)) ** 2 + (int(y2) - int(y1)) ** 2


def create_weighted_graph(x_coords, y_coords):
    #get the number of cities and initialize the graph
    num_cities = len(x_coords)
    graph = np.zeros((num_cities, num_cities))

    # Create a weighted graph from the x and y coordinates of the cities
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                graph[i, j] = calculate_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j])
            else:
                graph[i, j] = float('inf')

    return graph


def prims_algorithm(graph):
    
    # Get the number of vertices in the graph in the first row
    num_vertices = graph.shape[0]

    # Create an array to keep track of visited vertices, store the parent of each vertex in the MST, and the weight of edges
    visited = np.zeros(num_vertices, dtype=bool)
    parent = np.zeros(num_vertices, dtype=int)
    weight = np.full(num_vertices, float('inf'))

    # Start with the first vertex 
    weight[0] = 0
    parent[0] = -1

    for k in range(num_vertices):
        #initialize min weight and vertex as infinity
        min_weight = float('inf')
        min_vertex = -1
        # Find the vertex with the minimum weight that has not been visited
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



def find_perfect_matching(odd_vertices, matrix):
    # Initalize arrays for index and edges
    edges = []
    indexes = []

    # Find the minimum edge for each vertex
    for i in range(len(odd_vertices)):
        if i in indexes:
            continue
        # Initialize the minimum and index
        min = float('inf')
        min_index = -1

        # Find the minimum edge
        for j in range(len(odd_vertices)):
            if i != j and j not in indexes and matrix[odd_vertices[i]][odd_vertices[j]] < min:
                min = matrix[odd_vertices[i]][odd_vertices[j]]
                min_index = j
        # Add the indexes and edges
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


def christofides():
    #create a global adjencey matrix from the cities
    global graph
    graph = create_weighted_graph(x_points, y_points)

    #find the minimum spanning tree
    mst = prims_algorithm(graph)
    #find the odd degree nodes
    odd_degree_nodes = find_odd_degree_nodes(mst)

    #create pairs from the odd verticies and connect them
    matching = find_perfect_matching(odd_degree_nodes, graph)
    multiGraph = []

    #create a multigraph from the mst and the matching
    for i in range(len(mst)):
        multiGraph.append([i, mst[i]])

    for i in range(len(matching)):
        multiGraph[matching[i][0]].append(matching[i][1])

    #repsresnt the multigraph in nx
    nxMultiGraph = nx.MultiGraph()
    for edge in multiGraph[1:]:
        x = edge[0]
        for i in edge[1:]:
            nxMultiGraph.add_edge(x, i)

    #find the eulerian path
    eulerianPath = list(nx.eulerian_circuit(nxMultiGraph))

    tour = [0]
    #create the tour from the eulerian path by eliminatening the duplicates
    for (i, j) in eulerianPath:
        if j not in tour:
            tour.append(j)
    return tour

# File reading and assigning the values
start = time.time()
# We get the file name and read it here
with open(input("Enter a file name: "), 'r') as file:
    line = file.readline()
    # assigning the lists the values by using append
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

# get the size of the whole cities
size = int(digits[0]) + 1
# create an X array that contains the x and y points
X = np.array(listPoints, dtype='int')
#delete list points because it is not needed anymore
del listPoints
gc.collect()
# applying kmeans dynamically so we get the cluster size as element size / 750
cluster_size = math.ceil(size / 750)
# if cluster size is so big then we cannot handle the combinations and it will take unfeasible time, so we limit it
if cluster_size > 27:
    cluster_size = 27
# apply k means get the results and centroids, applied the kmeans++ version
kmeans = KMeans(n_clusters=cluster_size, init='k-means++', random_state=0, n_init='auto')
y = kmeans.fit_predict(X)
cent = kmeans.cluster_centers_

# if cluster size is bigger than 1 then we must select some of them
if cluster_size > 1:
    # density and number of cities lists created
    densities = []
    num_of_cities = []
    for pt in range(cluster_size):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        # get the elements of a cluster
        elements = np.array(X[y == pt].flatten())
        # find the density and append for that cluster
        densities.append(calculate_density(elements, cent[pt][0], cent[pt][1]))
        # also add it to the number of cities list
        num_of_cities.append(X[y == pt, 0].size)
        plt.scatter(X[y == pt, 0], X[y == pt, 1], s=1, color=[r, g, b])
        plt.scatter(cent[:, 0], cent[:, 1], s=20, c='yellow')
    # create them as a numpy array and make it a nx5 numpy matrix
    # centroid x-y-id-density-number of cities
    density_arr = np.array(densities)
    ids = np.arange(cluster_size)
    num_of_cities_array = np.array(num_of_cities)
    features = np.column_stack((ids, density_arr))
    clusters = np.hstack((cent, features))
    clusters_2 = np.column_stack((clusters, num_of_cities_array))
    # sort the list
    sort_list(clusters_2)
    # get the best list defined and create a boolean for sizing
    best_list = []
    acceptable_size = False
    cluster_size_2 = math.ceil(clusters_2.size / 10)
    # here we check if the chosen cluster satisfys the required city size, if not we try everything with one more cluster place in the list
    while not acceptable_size:
        for i in range(cluster_size_2): # setting the best and temp lists temporarly
            best_list = np.append(best_list, np.copy(clusters_2[i]), axis=0)
            temp_list = np.append(temp_list, np.copy(clusters_2[i]), axis=0)

        best_list = best_list.reshape(-1, 5)
        temp_list = temp_list.reshape(-1, 5)
        best_list = find_best(clusters_2, best_list, temp_list)
        if math.ceil(size / 2) < get_num_of_city(best_list): # checking if it satisfies the city size
            acceptable_size = True
        else: # if not repating with one more cluster space
            cluster_size_2 += 1
            best_list = []
            temp_list = []
            gc.collect()
    # terminating the clusters and cities until we reach exactly the wanted city size
    flat_xy_points = terminate_clusters(best_list, X, y)
    x_points = flat_xy_points[: math.ceil(size / 2)]
    y_points = flat_xy_points[math.ceil(size / 2):]
    plt.scatter(x_points, y_points, s=1, c='black')
    plt.scatter(best_list[:, 0], best_list[:, 1], s=20, c='red')
else:
    # from one cluster choose n/2 nodes
    plt.scatter(cent[:, 0], cent[:, 1], s=20, c='yellow')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], s=1, c='crimson')
    # get the whole size
    node_number = len(x_points)
    # create a lookup table and given values
    lookup = []
    cent_x = cent[0][0]
    cent_y = cent[0][1]
    # for each node in range get the distance and put it to the lookup table
    for pts in range(node_number):
        distance_btw_cent_and_node = (x_points[pts] - cent_x) ** 2 + (y_points[pts] - cent_y) ** 2
        lookup.append((pts, distance_btw_cent_and_node))
    # sort the lookup according to the distance
    lookup.sort(key=lambda a: a[1])
    needed_node = math.ceil(node_number / 2)
    temp_best_x = []
    temp_best_y = []
    # get the best using the lookup table and append it to the lists
    for ptr in range(needed_node):
        temp_best_x.append(x_points[lookup[ptr][0]])
        temp_best_y.append(y_points[lookup[ptr][0]])
    # assign them accordingly
    x_points = temp_best_x
    y_points = temp_best_y
    gc.collect()
    plt.scatter(x_points, y_points, s=5, c='blue')

# after the getting x and y values we create an id list for them and create it using a normal search
# it finds x and y values id's and we assume there is no duplicate points
node_number = len(x_points)
needed_node = math.ceil(size / 2)
id_points = []
for i in range(node_number):
    current_x = x_points[i]
    current_y = y_points[i]
    for j in range(size):
        if X[j, 0] == current_x and X[j, 1] == current_y:
            id_points.append(j)
            break

# for the duplicate ones we created a mini fix and update these here
duplicate_ids = []
duplicate_values = []
duplicate_found = False
for i in range(len(id_points)):
    for j in range(len(id_points)):
        if id_points[i] == id_points[j] and i is not j:
            duplicate_ids.append(j)
            duplicate_values.append(id_points[i])
            id_points[j] = id_points[i] + 1
            duplicate_found = True
            break
    if duplicate_found:
        break



# calling the tsp methods one by one
print("clustering done")
tour_christofides = christofides()
print("christofides done")
tour_optimized = three_opt(tour_christofides)
print("three opt done")
tour_optimized = two_opt(tour_optimized)
print("two opt done")
tour_optimized = two_opt(tour_optimized)
print("two opt done 2")
print(tour_optimized)
# get the correct ids for each id we got from the tsp functions
tour_optimized_ids = []
for i in range(len(tour_optimized)):
    tour_optimized_ids.append(id_points[tour_optimized[i]])
print(tour_optimized_ids)

for i in range(len(tour_optimized_ids)):
    plt.plot([X[tour_optimized_ids[i - 1], 0], X[tour_optimized_ids[i],0]], [X[tour_optimized_ids[i - 1],1],
        X[tour_optimized_ids[i],1]], 'r-',color=random.choice(['red', 'green', 'blue', 'yellow', 'black', 'purple', 'pink', 'orange']))
distance = get_length(tour_optimized)

# open an output file called output.txt and put the distance first then the ids
with open('output.txt', 'w') as output:
    output.write(str(distance) + '\n')
    for i in range(len(tour_optimized_ids)):
        output.write(str(tour_optimized_ids[i]) + '\n')

end = time.time()
elapsed_time = end - start
print('Execution time:', elapsed_time, 'seconds')
plt.show()
