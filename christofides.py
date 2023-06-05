import matplotlib.pyplot as plt
import numpy as np
import time
import networkx as nx
import random
import math


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
        
    plt.scatter(oddCoordsX, oddCoordsY, color='yellow')
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

nx.graph

def find_perfect_matching(odd_vertices, matrix):

    odd_vertices.sort(key=lambda v: sum(matrix[v]))
    edges= []
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

def create_minimum_weight_pairs(graph):
    minimum_weight_pairs = set()
    
    # Sort edges by weight in ascending order
    sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])

    # Create a set to keep track of used nodes
    used_nodes = set()

    for u, v, attr in sorted_edges:
        if u not in used_nodes and v not in used_nodes:
            minimum_weight_pairs.add((u, v))
            used_nodes.add(u)
            used_nodes.add(v)

    return minimum_weight_pairs


graph = create_weighted_graph(x_points, y_points)

parent = prims_algorithm(graph)



print(parent)
plot_mst(x_points, y_points, parent,[],[])

print("mst created")



odd_degree_nodes = find_odd_degree_nodes(parent)

print(odd_degree_nodes)

oddCoordsX = []
oddCoordsY = []
for i in range(len(odd_degree_nodes)):
        oddCoordsX.append(x_points[odd_degree_nodes[i]])
        oddCoordsY.append(y_points[odd_degree_nodes[i]])

plot_mst(x_points, y_points, parent,odd_degree_nodes,[])

print("odd degree nodes found")
#nxgraphOdd = nx.Graph(create_weighted_graph(oddCoordsX, oddCoordsY)*-1)

print("min odd graph created")
#matchingX = list(nx.max_weight_matching(nxgraphOdd, maxcardinality=True))

print(odd_degree_nodes)
matching = find_perfect_matching(odd_degree_nodes, graph)

#print("odd list count", nxgraphOdd.number_of_nodes())

print("matching size", len(odd_degree_nodes))

matching = list(create_minimum_weight_pairs(nx.Graph(create_weighted_graph(oddCoordsX, oddCoordsY))))

print("matching found")

matching = [(odd_degree_nodes[i],odd_degree_nodes[j]) for i,j in matching]
print(matching)

multiGraph = []
for i in range(len(parent)):
    multiGraph.append([i,parent[i]])

for i in range(len(matching)):
    multiGraph[matching[i][0]].append(matching[i][1])

plot_mst(x_points, y_points, [],odd_degree_nodes,matching)

print("matching added to multigraph")
print(multiGraph)

nxMultiGraph = nx.MultiGraph()

for edge in multiGraph[1:]:
    x = edge[0]
    for i in edge[1:]:
        nxMultiGraph.add_edge(x,i)

eulerianPath = list(nx.eulerian_circuit(nxMultiGraph))

#TODO create a multigraph from the minimum spanning tree and matching edges
#plot_mst(x_points, y_points, parent,odd_degree_nodes,matching)
#eulerianPath =generate_eulerian_circuit(multiGraph)
print("Eulerian Path")
print(eulerianPath)
tour=[0]
for(i,j) in eulerianPath:
    if j not in tour:
        tour.append(j)
print("Tour")
print(tour)
print("Tour sorted:")
print(sorted(tour))
dist = 0
for i in range(len(tour)):
    dist+= math.sqrt (calculate_distance(x_points[tour[i-1]], y_points[tour[i-1]], x_points[tour[i]], y_points[tour[i]]))

print("distance:", dist)
print("exec time:", time.time()-start)
#plot_Multi(x_points, y_points, multiGraph)
plot_Euler(x_points, y_points, tour)


