import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import time
import math
import gc


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

# File reading and assigning the values
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

    for i in range(cluster_size):
        r = np.round(np.random.rand(), 1)
        g = np.round(np.random.rand(), 1)
        b = np.round(np.random.rand(), 1)
        elements = np.array(X[y == i].flatten())
        densities.append(calculate_density(elements, cent[i][0], cent[i][1]))
        plt.scatter(X[y == i, 0], X[y == i, 1], s=1, color=[r, g, b])
        plt.scatter(cent[:, 0], cent[:, 1], s=20, c='yellow')
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
sort_list(clusters)
best_list = []
for i in range( int(cluster_size/2) ):
    best_list = np.append(best_list, np.copy(clusters[i]), axis = 0)
    temp_list = np.append(temp_list, np.copy(clusters[i]), axis = 0)

best_list = best_list.reshape(-1,4)
temp_list = temp_list.reshape(-1,4)
best_list = find_best(clusters, 0, best_list, 0)
print()
print(best_list)
plt.show()
end = time.time()
elapsed_time = end - start
print('Execution time:', elapsed_time, 'seconds')
