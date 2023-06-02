import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
import numpy as np
import time
import networkx as nx
import random


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
                graph[i,j] = calculate_distance(x_coords[i], y_coords[i], x_coords[j], y_coords[j])
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

def plot_mst(x_coords, y_coords, parent,odd,matching):
    plt.figure()
    for i in range(1, len(parent)):
        plt.plot([x_coords[i], x_coords[parent[i]]], [y_coords[i], y_coords[parent[i]]], 'r-')
    
    for i in range(len(matching)):
        plt.plot([x_coords[matching[i][0]], x_coords[matching[i][1]]], [y_coords[matching[i][0]], y_coords[matching[i][1]]], 'r-', color='purple')

    plt.scatter(x_coords, y_coords, color='b')
    
    oddCoordsX = []
    oddCoordsY = []
    for i in range(len(odd)):
        oddCoordsX.append(x_coords[odd[i]])
        oddCoordsY.append(y_coords[odd[i]])
        
    plt.scatter(oddCoordsX, oddCoordsY, color='g')
    plt.show()

def plot_Multi(x_coords, y_coords, parent):
    plt.figure()
    for i in range(1, len(parent)):
        for j in range(1,len(parent[i])):
            plt.plot([x_coords[i], x_coords[parent[i][j]]], [y_coords[i], y_coords[parent[i][j]]], 'r-')
    
    plt.scatter(x_coords, y_coords, color='b')

    plt.show()

def plot_Euler(x_coords, y_coords, parent):
    plt.figure()
    for i in range(len(parent)):
            plt.plot([x_coords[parent[i-1]], x_coords[parent[i]]], [y_coords[parent[i-1]], y_coords[parent[i]]], 'r-' ,color= random.choice(['red', 'green', 'blue', 'yellow', 'black', 'purple', 'pink', 'orange']))
    
    plt.scatter(x_coords, y_coords, color='b')

    plt.show()


def find_odd_degree_nodes(parent_data):
    # Initialize a dictionary to store the degree of each vertex
    degree = {}

    # Count the number of edges connected to each vertex
    for child, parent in enumerate(parent_data):
        if parent != -1:
            degree[parent] = degree.get(parent, 0) + 1
            degree[child] = degree.get(child, 0) + 1

    # Find the odd degree nodes
    odd_degree_nodes = [node for node, deg in degree.items() if deg % 2 != 0]

    return odd_degree_nodes






graph = create_weighted_graph(x_points, y_points)

parent = prims_algorithm(graph)

print("mst created")
odd_degree_nodes = find_odd_degree_nodes(parent)

oddCoordsX = []
oddCoordsY = []
for i in range(len(odd_degree_nodes)):
        oddCoordsX.append(x_points[odd_degree_nodes[i]])
        oddCoordsY.append(y_points[odd_degree_nodes[i]])

print("odd degree nodes found")
nxgraphOdd = nx.Graph(create_weighted_graph(oddCoordsX, oddCoordsY)*-1)

print("min odd graph created")
matchingX = list(nx.max_weight_matching(nxgraphOdd, maxcardinality=True))

print("odd list count", nxgraphOdd.number_of_nodes())
matching = []

for i in range(len(matchingX)):
    matching.append((odd_degree_nodes[matchingX[i][0]], odd_degree_nodes[matchingX[i][1]]))

print(matching)
print(parent)

multiGraph = []
for i in range(len(parent)):
    multiGraph.append([i,parent[i]])

for i in range(len(matching)):
    multiGraph[matching[i][0]].append(matching[i][1])
print
print(multiGraph)

nxMultiGraph = nx.MultiGraph()

for edge in multiGraph[1:]:
    x = edge[0]
    for i in edge[1:]:
        nxMultiGraph.add_edge(x,i)

eulerianPath = list(nx.eulerian_circuit(nxMultiGraph))

print(nxMultiGraph.edges)
print("Eulerian Path")
print(eulerianPath)
#TODO create a multigraph from the minimum spanning tree and matching edges
#plot_mst(x_points, y_points, parent,odd_degree_nodes,matching)

tour=[0]
for(i,j) in eulerianPath:
    if j not in tour:
        tour.append(j)
print("Tour")
print(tour)

print("exec time:", time.time()-start)
#plot_Multi(x_points, y_points, multiGraph)
plot_Euler(x_points, y_points, tour)