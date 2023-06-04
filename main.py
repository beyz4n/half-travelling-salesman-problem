import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
import math
import gc
import random
from itertools import combinations


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

    for i in range( int( cluster_size) ):
        j = i
        temp = list[i,1]
        index = i
        while(j+1< int( cluster_size) ):
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
        temp = list[i,4]
        list[i,4] = list[index,4]
        list[index, 4] = temp

    i = 1
    counter = 0
    while(i< int(cluster_size)):
        if(list[i-1,0] == list[i,0]):
            j = i
            while(j < int(cluster_size) and list[j-1,0] == list[j,0] ):
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
                temp = list[j,4]
                list[j,4] = list[index,4]
                list[index,4] = temp
                j += 1
        i += 1

    print()
    print(list)
    return list


temp_list = [] # use to store temp half lists
def find_best(elm, new_best_list, temp_list):
    cluster_size = math.ceil(elm.size/5)
    arr = range(cluster_size)
    best_cluster_size = math.ceil(new_best_list.size/5)
    combs = list(combinations(arr, best_cluster_size))
    #print(combinations)
    i = 0
    combs_size = len(combs)
    best_weight = get_weight(new_best_list)
    while(i<combs_size):
        
        if(math.ceil(size/2) < get_num_of_city(temp_list)):
            if(get_weight(temp_list) < best_weight):
                new_best_list = temp_list
                best_weight = get_weight(temp_list)

        temp_list = []
        temp_list = np.array(elm[combs[i],:])
        i += 1

    gc.collect()
    return new_best_list



# def find_best(elm, index, new_best_list, prev_index):
#     list = elm
#     i = index
#     while(i<= cluster_size ):
#         if( i + (cluster_size_2-index) <= cluster_size and index < i and prev_index < i ):
#             temp_list[index-1, 0] = list[i-1,0]
#             temp_list[index-1, 1] = list[i-1,1]
#             temp_list[index-1, 2] = list[i-1,2]
#             temp_list[index-1, 3] = list[i-1,3]
#             temp_list[index-1, 4] = list[i-1,4]
#         if(index < cluster_size_2 and  i + (cluster_size_2 - index) < cluster_size and prev_index < i ):
#             index += 1
#             prev_index = i
#             new_best_list = find_best(list, index, new_best_list, prev_index)
#             index -= 1
#             if(index == 0):
#                 return new_best_list

#         if(index == cluster_size_2 and prev_index < i ):
#             if( get_weight(temp_list) < get_weight(new_best_list) and math.ceil(size/2) <= get_num_of_city(temp_list) ):
#                 new_best_list = temp_list
#         i += 1
#     return new_best_list

def get_num_of_city(elm):
    num = 0
    i = 0
    elm_size = int(elm.size/5)
    while(i < elm_size):
        num += elm[i, 4]
        i += 1
    return num
    
# def get_weight(list):
#     weight = 0
#     temp_density = list[0,3]
#     temp_density_2 = 0
#     i = 1
#     list_size = int(list.size/5)
#     while(i < list_size ):
#         temp_density_2 = list[i,3]
#         weight += math.sqrt( (list[i,0]-list[i-1,0])**2 +  (list[i,1]-list[i-1,1])**2 ) * temp_density_2 * temp_density
#         temp_density = temp_density_2
#         i += 1
    
    
#     return weight

def get_weight(elm):
    weight = 0
    temp_elm = np.copy(elm)
    i = 0
    elm_size = int(temp_elm.size/5)
    best_distance = math.sqrt((temp_elm[0,0]-temp_elm[1,0])**2 + (temp_elm[0,1]-temp_elm[1,1])**2)
    temp_distance = 0
    index = 0
    next_index = 0
    if(elm_size == 1):
        weight += temp_elm[0,3]
    
    while(elm_size != 1):
        
        while(i<elm_size):
            if(i != index):
                temp_distance = math.sqrt((temp_elm[index,0]-temp_elm[i,0])**2 + (temp_elm[index,1]-temp_elm[i,1])**2)
                if(temp_distance<best_distance):
                    best_distance = temp_distance
                    next_index = i
            i += 1
        i = 0
        weight += best_distance * temp_elm[index,3]*temp_elm[next_index,3]
        temp_elm = np.delete(temp_elm, index,0)
        index = next_index-1
        next_index = 0
        best_distance = math.sqrt((temp_elm[0,0]-temp_elm[index,0])**2 + (temp_elm[0,1]-temp_elm[index,1])**2)
        elm_size = int(temp_elm.size/5)
    
    return weight

def terminate_clusters(best_cluster, X, y):
    print()
    print(best_cluster)
    
    best_cluster_size = int(best_cluster.size/5)
    i = 0
    center_X = 0
    center_Y = 0
    city_size = get_num_of_city(best_cluster)
    while(i<best_cluster_size):
        center_X += best_cluster[i,0]*best_cluster[i,4]
        center_Y += best_cluster[i,1]*best_cluster[i,4]
        i+=1
    center_X =  center_X/city_size
    center_Y = center_Y/city_size

    temp_X = 0
    temp_Y = 0
    i = 0
    x_axis = np.copy(X[y==best_cluster[0,2], 0].flatten())
    y_axis = np.copy(X[y==best_cluster[0,2], 1].flatten())
    for i in range(1, best_cluster_size):
        x_axis = np.append(x_axis, X[y==best_cluster[i,2], 0].flatten())
        y_axis = np.append(y_axis, X[y==best_cluster[i,2], 1].flatten())
    
    
    x_axis = x_axis.flatten()
    y_axis = y_axis.flatten()


    x_point_length = len( x_axis )

    for i in range( int( x_point_length) ):
        j = i
        temp_X = x_axis[0]
        temp_Y = y_axis[0]
        index = i
        while( j+1< x_point_length ):
            if( (y_axis[j+1] - center_Y)**2 + (x_axis[j+1] - center_X)**2 < (temp_X - center_X)**2 + (temp_Y - center_Y)**2 ):
                temp_X = x_axis[j+1]
                temp_Y = y_axis[j+1]
                index = j+1
            j += 1

        temp = x_axis[i]
        x_axis[i] = x_axis[index]
        x_axis[index] = temp
        temp = y_axis[i]
        y_axis[i] = y_axis[index]
        y_axis[index] = temp
    
    min_city_size = math.ceil(size/2)
    temp_X_2 =[]
    temp_Y_2 = []
    i = 0
    for i in range(min_city_size):
        temp_X_2.append(x_axis[i])
        temp_Y_2.append(y_axis[i])
    
    flat_xy = temp_X_2+temp_Y_2
    
    return flat_xy

    return
    # distance_table = []
    # i = 0
    # for i in range(point_length):
    #     distance_table.append((i, (y_axis[i] - center_Y)**2 + (x_axis[i] - center_X)**2  ))
    
    # distance_table.sort(key=lambda a: a[1])
    # min_city_size = math.ceil(size/2)

    # i = 0
    # for i in range(min_city_size):
    #     temp_X.append(x_points[distance_table[i][0]])
    #     temp_Y.append(y_points[ distance_table[i][0] ])
    
    # flat_xy = temp_X+temp_Y
    
    # return flat_xy

    # for i in range( int( best_cluster_size) ):
    #     j = i
    #     temp_X = best_cluster[i,0]
    #     temp_Y = best_cluster[i,1]
    #     index = i
    #     while( j+1< best_cluster_size ):
    #         if( (best_cluster[j+1,1] - center_Y)**2 + (best_cluster[j+1,0] - center_X)**2 < (temp_X - center_X)**2 + (temp_Y - center_Y)**2 ):
    #             temp_X = best_cluster[j+1,0]
    #             temp_Y = best_cluster[j+1,1]
    #             index = j
    #         j += 1

    #     temp = best_cluster[i,1]
    #     best_cluster[i,1] = best_cluster[index,1]
    #     best_cluster[index, 1] = temp
    #     temp = best_cluster[i,0]
    #     best_cluster[i,0] = best_cluster[index,0]
    #     best_cluster[index, 0] = temp
    #     temp = best_cluster[i,2]
    #     best_cluster[i,2] = best_cluster[index,2]
    #     best_cluster[index, 2] = temp
    #     temp = best_cluster[i,3]
    #     best_cluster[i,3] = best_cluster[index,3]
    #     best_cluster[index, 3] = temp
    #     temp = best_cluster[i,4]
    #     best_cluster[i,4] = best_cluster[index,4]
    #     best_cluster[index, 4] = temp

    # best_cluster_size = best_cluster.size/5
    # i = int(best_cluster_size-1)
    # needed_city_size = math.ceil(size/2)
    # while(needed_city_size<city_size):
    #     while(0<=i):
    #         if(needed_city_size <= city_size - best_cluster[i,4]):
    #             best_cluster = np.delete(best_cluster, i,0)
                
    #         else:
    #             print("moin")
    #             # cluster icini sortla ve en uzaklari cikarmaya basla

    #         best_cluster_size = best_cluster.size/5
    #         i -= 1
    #         city_size = get_num_of_city(best_cluster)
    #     break
    
    #return 

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
    num_of_cities =[]
    for pt in range(cluster_size):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        elements = np.array(X[y == pt].flatten())
        densities.append(calculate_density(elements, cent[pt][0], cent[pt][1]))
        num_of_cities.append(X[y== pt, 0].size)
        plt.scatter(X[y == pt, 0], X[y == pt, 1], s=1, color=[r, g, b])
        plt.scatter(cent[:, 0], cent[:, 1], s=20, c='yellow')
        # pearson_corr.append(pearsonr(X[y==i, 0],X[y==i, 1]))
        # norms.append(stats.norm.pdf(X))
        # scipy.stats.multivariate_normal()
    density_arr = np.array(densities)
    ids = np.arange(cluster_size)
    num_of_cities_array = np.array(num_of_cities)
    features = np.column_stack((ids, density_arr))
    clusters = np.hstack((cent, features))
    clusters_2 = np.column_stack((clusters, num_of_cities_array))
    print(clusters_2)
    sort_list(clusters_2)
    best_list = []
    acceptable_size = False
    j = 0
    cluster_size_2 = math.ceil(clusters_2.size/10)
    while(not acceptable_size):
        for i in range(cluster_size_2 + j):
            best_list = np.append(best_list, np.copy(clusters_2[i]), axis=0)
            temp_list = np.append(temp_list, np.copy(clusters_2[i]), axis=0)

        best_list = best_list.reshape(-1, 5)
        temp_list = temp_list.reshape(-1, 5)
        best_list = find_best(clusters_2, best_list, temp_list)
        if(math.ceil(size/2)<get_num_of_city(best_list)):
            acceptable_size = True
        else :
            cluster_size_2 += 1
            j += 1    
            best_list = []
            temp_list = []
            gc.collect()
    
    print()
    print(best_list)
    print(get_num_of_city(best_list))
    flat_xy_points = terminate_clusters(best_list,X,y)
    x_points = flat_xy_points[: math.ceil(size/2)]
    y_points = flat_xy_points[math.ceil(size/2) :]
    print("lengths: ")
    print(len(x_points))
    print(len(y_points))
    print(len(flat_xy_points))
    plt.scatter(x_points, y_points, s=1, c='black')
    plt.scatter(best_list[:,0], best_list[:, 1], s=20, c='red')
else:
    # from one cluster choose n/2 nodes
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
