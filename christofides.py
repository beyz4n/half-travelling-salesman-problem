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



def calculate_distance(x1, y1, x2, y2):
    return (x2 - x1)**2 + (y2 - y1)**2

def create_weighted_graph(x_coords, y_coords):
    num_cities = len(x_coords)
    graph = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance = calculate_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j])
                graph[i, j] = distance
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

    for _ in range(num_vertices):
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

def plot_mst(x_coords, y_coords, parent):
    plt.figure()
    for i in range(1, len(parent)):
        plt.plot([x_coords[i], x_coords[parent[i]]], [y_coords[i], y_coords[parent[i]]], 'r-')
    plt.scatter(x_coords, y_coords, color='b')
    plt.show()


graph = create_weighted_graph(x_points, y_points)

parent = prims_algorithm(graph)
plot_mst(x_points, y_points, parent)
