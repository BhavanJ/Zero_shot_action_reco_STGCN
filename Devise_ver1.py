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

plt.ion()
use_cuda = torch.cuda.is_available()

#TODO: Verify the below thing about unseen classes
UNSEEN_CLASSES = [3,8,11,16,51]
DIM_STGCN = 256
DIM_LANGAUGE = 600
MARGIN = 0.1
N_EPOCH = 10
NO_CLASS = 60
BATCH_SIZE = 64

FEATURE_DIR = './work_dir/NTU-RGB-D_zero_shot/xview/ST_GCN_test/'
SKELETON_DIR = './data/NTU-RGB-D_zero_shot/xview/'

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

		margin_var = Variable(torch.ones(batch_sz, 1,)*self.margin)
		margin_var = margin_var.cuda() if use_cuda else margin_var

		zero_var = Variable(torch.zeros(batch_sz, 1,))
		zero_var = zero_var.cuda() if use_cuda else zero_var

		#loss_val = 0
		for cls_no in range(NO_CLASS):

			# if cls_no == class_label:
			# 	#loss_val +=0
			# 	continue
			# else:
			# 	# loss_val += torch.max(0,self.margin-true_embedding*vision_feature_projected+langauge_embeddings[cls_no]*vision_feature_projected)
			# 	# loss_val += torch.max(0, self.margin - (true_embedding*vision_feature_projected) + (langauge_embeddings[cls_no].view(1,-1).repeat(batch_sz,1)*vision_feature_projected)  )
			# 	loss_val += torch.sum(torch.max(zero_var, margin_var - (true_embedding*vision_feature_projected) + (langauge_embeddings[cls_no].view(1,-1).repeat(batch_sz,1)*vision_feature_projected)  ) )

			loss_val += torch.sum(torch.max(zero_var, margin_var - (true_embedding*vision_feature_projected) + (langauge_embeddings[cls_no].view(1,-1).repeat(batch_sz,1)*vision_feature_projected)  ) )

		pdb.set_trace()
		
		return loss_val/batch_sz    



class DataLoaderSTGCN(Dataset):
	def __init__(self,train_mode):

		if train_mode == True:
			self.features_array =  np.load(FEATURE_DIR+'features_array_train.npy')
			self.labels_array   =  np.load(FEATURE_DIR+'labels_array_train.npy')
		if train_mode == False:
			self.features_array =  np.load(FEATURE_DIR+'features_array_test.npy')
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

##################################################################################################################

train_dataset = DataLoaderSTGCN(train_mode = True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataset = DataLoaderSTGCN(train_mode = False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

net = Devise(in_dim = DIM_STGCN,out_dim = DIM_LANGAUGE)
if use_cuda:
	net = net.cuda()
print(net)

criterion = Custom_devise_loss(margin = MARGIN)
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
		projected_STGCN_feature = net(STGCN_feature)
		#language_feature = langauge_embeddings[class_label.view(-1)]	

		#pdb.set_trace()

		##TODO...need to set the grad_func for loss to true
		loss = criterion(projected_STGCN_feature,langauge_embeddings,class_label)
		loss.backward()
		optimizer.step()
		running_loss += loss.data

	print('Epoch: ' + str(epoch) + '  Loss:' + str(running_loss))    

print('Finished Training')
pdb.set_trace()

torch.save(net.state_dict(), MODEL_PATH)


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




