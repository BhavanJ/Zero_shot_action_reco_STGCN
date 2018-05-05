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
def get_furthest_n(one_example):
    nbrs = NearestNeighbors(n_neighbors=60, algorithm='auto', metric='cosine').fit(Full_data) # pass one training example here
    distances, nn_indices = nbrs.kneighbors(one_example)
#     print('nearest neighbours:')
#     for i, index in enumerate(nn_indices[0][1:]):
#         print(classnames[index], distances[0][i])
#     print('\n')
    return nn_indices[0][-1], distances[0][-1] # get last element
    
    
# numpy.argsort
# this is unsorted
a = np.array((60,1))
arr_dist = []
arr_nn = []
for i in range(len(classnames)):
    one_em = bigram_embeddings[i][:]
    nn_ind, dist = get_furthest_n(one_em.reshape(1,-1))
    #print(classnames[i], float(dist), str(classnames[int(nn_ind)])) # unsorted
    arr_dist.append(float(dist))
    arr_nn.append(classnames[int(nn_ind)])
# print(arr_dist)
# print(arr_nn)


b = np.asarray(arr_dist)
# unsorted

c = list(np.argsort(b)) # this is sorted in ascending order

print(c)


for class_n in c:
    print(classnames[class_n], arr_dist[class_n], arr_nn[class_n])

# these are the furthest neighbors for each class ordered from closest to furthest