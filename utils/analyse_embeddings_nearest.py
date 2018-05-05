from sklearn.neighbors import NearestNeighbors
import numpy as np
import torch


# get classnames
classnames = []
with open('action_classes_original.txt', 'r') as f:
    classnames = f.read().split('\n')
    for i in f:
        classnames.append(i)
        
        
# get embeddings
bigram_embeddings = np.load('bigram_embeddings.npy')



# get neighbors
Full_data = bigram_embeddings
def get_nearest_n(one_example):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto', metric='cosine').fit(Full_data) 
    distances, nn_indices = nbrs.kneighbors(one_example)
#     print('nearest neighbours:')
#     for i, index in enumerate(nn_indices[0][1:]):
#         print(classnames[index], distances[0][i])
#     print('\n')
    return nn_indices[0][1:], distances[0][1:] # nn_indices is a array of length 2
    #print classnames[nn_indices]

# numpy.argsort
# this is unsorted
a = np.array((60,1))
arr_dist = []
arr_nn = []
for i in range(len(classnames)):
    one_em = bigram_embeddings[i][:]
    nn_ind, dist = get_nearest_n(one_em.reshape(1,-1))
    #print(classnames[i], float(dist), str(classnames[int(nn_ind)]))
    arr_dist.append(float(dist))
    arr_nn.append(classnames[int(nn_ind)])
# print(arr_dist)
# print(arr_nn)

b = np.asarray(arr_dist)

c = list(np.argsort(b)) # this is sorted in ascending order

print(c)

for class_n in c:
    print(classnames[class_n], arr_dist[class_n], arr_nn[class_n])

# these are the most distinct classes in the set # sorted in ascending order