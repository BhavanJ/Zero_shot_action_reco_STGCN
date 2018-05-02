from __future__ import print_function, division
import os
import time
import numpy as np
import yaml
import pickle
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pdb
import torch.nn.functional as F
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

#plt.ion()
use_cuda = torch.cuda.is_available()
print('lr=0.001, test_batch=64, annealling, neighbour_loss, 100 epochs, Adam', "UNSEEN_CLASSES_ONLY") # change test_batch = 32 for 70% at 60 epochs
#TODO: for loss function u MIGHT WANT TO SET grad function to zero
#TODO: Verify the below thing about unseen classes
UNSEEN_CLASSES = [3,8,11,16,51]
DIM_STGCN = 256
DIM_LANGAUGE = 600
MARGIN = 0.1
N_EPOCH = 100
NO_CLASS = 60
BATCH_SIZE = 64
LR = 0.001
MOMENTUN = 0.9


NORMALIZE_VIS = True
NORMALIZE_EMB = True

#FEATURE_DIR = './work_dir/NTU-RGB-D_zero_shot/xview/ST_GCN_test/'
FEATURE_DIR = './features/'
#SKELETON_DIR = './data/NTU-RGB-D_zero_shot/xview/'
MODEL_PATH = './models/model'
##Loading all the files
# features_array_test = np.load(FEATURE_DIR+'features_array_test.npy')
# features_array_train = np.load(FEATURE_DIR+'features_array_train.npy')
# labels_array_test = np.load(FEATURE_DIR+'labels_array_test.npy')
# labels_array_train = np.load(FEATURE_DIR+'labels_array_train.npy')
langauge_embeddings = np.load(FEATURE_DIR+'class_embeddings_temp.npy')


## TODO: MAKE THESE UNIT NORM.......IMP
#features_array_test
#features_array_train

class Devise(nn.Module):
	def __init__(self,in_dim,out_dim):
		super(Devise, self).__init__()
		self.fc1 = nn.Linear(in_dim,out_dim)
		
	def forward(self, vision_feature):#,language_feature,mode='train'):
		x = self.fc1( vision_feature )
		return x

class Custom_devise_loss(nn.Module):
	def __init__(self,margin):
		super(Custom_devise_loss,self).__init__()
		self.margin = margin

	def forward(self,vision_feature_projected,langauge_embeddings,class_label):	

		# pdb.set_trace()
		true_embedding = langauge_embeddings[class_label.view(-1)]
		#_assert_no_grad(class_label) #TODO: VERIFY THIS

		batch_sz = vision_feature_projected.shape[0]

		margin_var = Variable(torch.ones(batch_sz)*self.margin)
		margin_var = margin_var.cuda() if use_cuda else margin_var

		zero_var = Variable(torch.zeros(batch_sz))
		zero_var = zero_var.cuda() if use_cuda else zero_var

		loss_val = 0
		for cls_no in range(NO_CLASS):

			# if cls_no == class_label:
			# 	#loss_val +=0
			# 	continue
			# else:
			# 	# loss_val += torch.max(0,self.margin-true_embedding*vision_feature_projected+langauge_embeddings[cls_no]*vision_feature_projected)
			# 	# loss_val += torch.max(0, self.margin - (true_embedding*vision_feature_projected) + (langauge_embeddings[cls_no].view(1,-1).repeat(batch_sz,1)*vision_feature_projected)  )
			# 	loss_val += torch.sum(torch.max(zero_var, margin_var - (true_embedding*vision_feature_projected) + (langauge_embeddings[cls_no].view(1,-1).repeat(batch_sz,1)*vision_feature_projected)  ) )

			loss_val += torch.sum(torch.max(zero_var, margin_var - torch.sum((true_embedding*vision_feature_projected),dim=1) + torch.sum((langauge_embeddings[cls_no].view(1,-1).repeat(batch_sz,1)*vision_feature_projected),dim=1)  ),dim=0 )
			#pdb.set_trace()	
	
		loss_val = loss_val/batch_sz

		return loss_val

class DataLoaderSTGCN(Dataset):
	def __init__(self,train_mode):

		if train_mode == True:

			self.features_array =  np.load(FEATURE_DIR+'features_array_train.npy')
			if NORMALIZE_VIS == True:
				self.features_array = preprocessing.normalize(self.features_array, axis=1, copy=False)	
			self.labels_array   =  np.load(FEATURE_DIR+'labels_array_train.npy')

			#norm = np.linalg.norm(self.features_array,ord = 1, axis = 1)
 			#np.sqrt(np.sum((self.features_array[0])*(self.features_array[0])))

		if train_mode == False:
			
			self.features_array =  np.load(FEATURE_DIR+'features_array_test.npy')
			if NORMALIZE_VIS == True:
				self.features_array = preprocessing.normalize(self.features_array, axis=1, copy=False)	
			self.labels_array   =  np.load(FEATURE_DIR+'labels_array_test.npy')

		#self.langauge_embeddings  =	 np.load(FEATURE_DIR+'class_embeddings_temp.npy')	

		self.features_array = torch.FloatTensor(self.features_array)
		self.features_array = self.features_array.cuda() if use_cuda else self.features_array
		self.labels_array = torch.LongTensor(self.labels_array)
		self.labels_array = self.labels_array.cuda() if use_cuda else self.labels_array

	def __len__(self):	
		return self.features_array.shape[0]

	def __getitem__(self, idx):

		sample = {
		'feature' : self.features_array[idx],
		'label'   : self.labels_array[idx:idx+1]
		}
		return sample


def get_nn(classnames, Full_data, one_example):
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto', metric='cosine').fit(Full_data) # pass one training example here
    distances, nn_indices = nbrs.kneighbors(one_example)
    # print('nearest neighbours:')
    # for i, index in enumerate(nn_indices[0][1:]):
    #     print(classnames[index], distances[0][i])
    # print('\n')

    return nn_indices[0][0], distances # nn_indices is a array of length 3
    #print classnames[nn_indices]
	
	

def top_k_accuracy(k=3, only_unseen=False):
	# Uses nearest neighbours 
	if only_unseen:
		embeddings = langauge_embeddings[UNSEEN_CLASSES].data.numpy()
	else:
		embeddings = langauge_embeddings.data.numpy()


	nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='cosine').fit(embeddings) 


	correct = 0.0
	total = 0.0
	for batch_idx,batch_data in enumerate(test_dataloader):

		STGCN_feature = Variable(batch_data['feature']) 
		class_label = Variable(batch_data['label'])
		projected_STGCN_feature = net(STGCN_feature)


		if only_unseen:
			_, unseen_index = nbrs.kneighbors(projected_STGCN_feature.data)
			#print(unseen_index[0])
			#predicted_label = UNSEEN_CLASSES[unseen_index[0]]
			predicted_label = [UNSEEN_CLASSES[unseen_index[0][x]] for x in range(k)] # or in range k 
			#print('class_label', class_label)

		else:
			_, predicted_cls = nbrs.kneighbors(projected_STGCN_feature.data)
			predicted_label = predicted_cls[0]


		#if predicted_label == class_label.data[0][0]:
		if class_label.data[0][0] in set(predicted_label):
			correct+=1.0
		total+=1.0
		# pdb.set_trace()

	acc = correct/total*100.0
	return acc	




def test_accuracy(only_unseen=False):

	if only_unseen:
		embeddings = langauge_embeddings[UNSEEN_CLASSES].data.numpy()
	else:
		embeddings = langauge_embeddings.data.numpy()

	
	nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto', metric='cosine').fit(embeddings) # try with kd_tree later	

	correct = 0.0
	total = 0.0
	for batch_idx,batch_data in enumerate(test_dataloader):

		STGCN_feature = Variable(batch_data['feature']) 
		class_label = Variable(batch_data['label'])
		projected_STGCN_feature = net(STGCN_feature)


		######## Compute predicted class using dot product	

		#_,predicted_cls = torch.max(torch.sum(embeddings*projected_STGCN_feature,dim=1),dim=0)


		##### Compute predicted class using nearest neighbours 

		if only_unseen:
			_, unseen_index = nbrs.kneighbors(projected_STGCN_feature.data)
			#print(unseen_index[0][0])
			predicted_label = UNSEEN_CLASSES[unseen_index[0][0]]

		
		else:
			_, predicted_cls = nbrs.kneighbors(projected_STGCN_feature.data)
			predicted_label = predicted_cls[0][0]


		if predicted_label == class_label.data[0][0]:

		#if predicted_cls.data[0] == class_label.data[0][0]:
			correct+=1.0
		total+=1.0
		# pdb.set_trace()

	acc = correct/total*100.0
	return acc	
	
##################################################################################################################



langauge_embeddings = np.load(FEATURE_DIR+'class_embeddings_temp.npy')


train_dataset = DataLoaderSTGCN(train_mode = True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataset = DataLoaderSTGCN(train_mode = False)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print('test_dataset', len(test_dataset))


net = Devise(in_dim = DIM_STGCN,out_dim = DIM_LANGAUGE)
if use_cuda:
	net = net.cuda()
print(net)

criterion = Custom_devise_loss(margin = MARGIN)
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUN)
optimizer = optimizer = optim.Adam(net.parameters(), lr=LR)
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCH)


if NORMALIZE_EMB == True:
	# pdb.set_trace()
	# norm = np.linalg.norm(langauge_embeddings,ord = 1, axis = 1).reshape(60,1)
	# langauge_embeddings = langauge_embeddings/norm	
	langauge_embeddings = preprocessing.normalize(langauge_embeddings, axis=1, copy=False)

langauge_embeddings = Variable(torch.FloatTensor(langauge_embeddings))
langauge_embeddings = langauge_embeddings.cuda() if use_cuda else langauge_embeddings




for epoch in range(N_EPOCH):  

	running_loss = 0.0
#	for idx, data in enumerate(train_loader,0):
	for batch_idx,batch_data in enumerate(train_dataloader):

		optimizer.zero_grad()
		#print(batch_idx, batch_data['feature'].shape,batch_data['label'].shape)
		STGCN_feature = Variable(batch_data['feature']) 
		# STGCN_feature = STGCN_feature.cuda() if use_cuda else STGCN_feature
		class_label = Variable(batch_data['label'])
		# class_label = class_label.cuda() if use_cuda else class_label				
		projected_STGCN = net(STGCN_feature)
		#language_feature = langauge_embeddings[class_label.view(-1)]	


		projected_STGCN_feature = F.normalize(projected_STGCN, p=2, dim=1)
		#pdb.set_trace()

		# if epoch == 0:

		# 	acc = test_accuracy()
		# 	print('baseline=', acc)


		##TODO...need to set the grad_func for loss to true
		loss = criterion(projected_STGCN_feature,langauge_embeddings,class_label)
		loss.backward()
		optimizer.step()
		running_loss += loss.data[0]

	running_loss = running_loss/len(train_dataloader)
	
	top_k = top_k_accuracy(k=3, only_unseen=False)

	accuracy = test_accuracy()
	# pdb.set_trace()
	print('Epoch: ' + str(epoch) + '  Train Loss: ' + str(running_loss) + '  Test Accuracy: ' + str(accuracy) + ' Top k Accuracy: ', str(top_k))    

print('Finished Training')
#pdb.set_trace()

#torch.save(net.state_dict(), MODEL_PATH)





####################################################################################################

#feature_variable = Variable(torch.FloatTensor(feature_data_stacked_train))
#feature_variable = feature_variable.cuda() if use_cuda else feature_variable
#label_variable = Variable(torch.LongTensor(true_label_stacked_train))
#label_variable = label_variable.cuda() if use_cuda else label_variable

# skeleton_data_train  =  np.load(SKELETON_DIR+'train_data.npy')
# skeleton_label_train =  pickle.load( open( SKELETON_DIR+'train_label.pkl', "r" ) )
# skeleton_data_test   =  np.load(SKELETON_DIR+'val_data.npy')
# skeleton_label_test  =  pickle.load( open( SKELETON_DIR+'val_label.pkl', "r" ) )

# features_array_train.shape
# labels_array_train.shape
# skeleton_data_train.shape
# len(skeleton_label_train[0])

# features_array_test.shape
# labels_array_test.shape
# skeleton_data_test.shape
# len(skeleton_label_test[0])

# Test set  --> 22051
# Train set --> 34527
# 22051.0/(22051.0+34527.0)*100.0 --> 38.97%

# langauge_embeddings.shape



# if __name__ == '__main__':
# 	print('lr=0.01')
# 	main()
