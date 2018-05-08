import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import scipy.io as sio
import math
import argparse
import random
import os
from sklearn.metrics import accuracy_score
import time
import yaml
import pickle
from collections import OrderedDict
from sklearn import preprocessing
import pdb

# Written by Bhavan Jasani (Maverick)
# Based on https://github.com/lzrobots/LearningToCompare_ZSL

#TODO:.....add if use_GPU, normalize the embeddings, add unseen classes
##where are they converting tensors to Variables

FEATURE_DIR = '/home/bjasani/Desktop/CMU_HW/VLR_project/STGN/st-gcn/work_dir/NTU-RGB-D_zero_shot_furtherest_neigh/xview/ST_GCN_exp1_features/'
SKELETON_DIR = '/home/bjasani/Desktop/CMU_HW/VLR_project/STGN/st-gcn/data/NTU-RGB-D_zero_shot_furtherest_neigh/xview/'
FEATURES_TRAIN_NAME 	= 'features_array_train_exp_1.npy'
FEATURES_TEST_NAME  	= 'features_array_test_exp_1.npy'
LABELS_TRAIN_NAME       = 'labels_array_train_exp_1.npy'
LABELS_TEST_NAME    	= 'labels_array_test_exp_1.npy'
LANGUAGE_EMBEDDING_NAME = 'bigram_embeddings.npy'


NO_CLASSES 			= 60 
NO_SEEN_CLASSES 	= 55
NO_UNSEEN_CLASSES 	= 5
UNSEEN_CLASSES = [10,24,41,52,55] # For furthest neighbhours...classes go from 0 to 59 here 
DIM_STGCN = 256
DIM_LANGAUGE = 700
NORMALIZE_VIS = True
NORMALIZE_EMB = True
LANG_EMB_RANDOM = False
BATCH_SIZE = 32
EPISODE = 500000
TEST_EPISODE = 1000
LEARNING_RATE = 1e-5
GPU = 0

#use_cuda = torch.cuda.is_available()

print('############################')
print('############################')
print('LANGUAGE_EMBEDDING_RANDOM: ', LANG_EMB_RANDOM)
print('############################')
print('############################')

class AttributeNetwork(nn.Module):
	"""docstring for RelationNetwork"""
	def __init__(self,input_size,hidden_size,output_size):
		super(AttributeNetwork, self).__init__()
		self.fc1 = nn.Linear(input_size,hidden_size)
		self.fc2 = nn.Linear(hidden_size,output_size)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		return x

class RelationNetwork(nn.Module):
	"""docstring for RelationNetwork"""
	def __init__(self,input_size,hidden_size,):
		super(RelationNetwork, self).__init__()
		self.fc1 = nn.Linear(input_size,hidden_size)
		self.fc2 = nn.Linear(hidden_size,1)

	def forward(self,x):
		x = F.relu(self.fc1(x))
		x = F.sigmoid(self.fc2(x))
		return x


def main():
	print("init dataset")

	train_features =  np.load(FEATURE_DIR+FEATURES_TRAIN_NAME)
	if NORMALIZE_VIS == True:
		train_features = preprocessing.normalize(train_features, axis=1, copy=False)	
	train_label   =  np.load(FEATURE_DIR+LABELS_TRAIN_NAME).astype(int)

	train_features = torch.from_numpy(train_features)
	train_label=torch.from_numpy(train_label).unsqueeze(1)


	langauge_embeddings = np.load(FEATURE_DIR+LANGUAGE_EMBEDDING_NAME)
	if LANG_EMB_RANDOM == True:
		langauge_embeddings = np.random.random(langauge_embeddings.shape)

	if NORMALIZE_EMB == True:
		# norm = np.linalg.norm(langauge_embeddings,ord = 1, axis = 1).reshape(60,1)
		# langauge_embeddings = langauge_embeddings/norm	
		langauge_embeddings = preprocessing.normalize(langauge_embeddings, axis=1, copy=False)


	all_attributes = langauge_embeddings
	attribute = langauge_embeddings
	testclasses_id = np.asarray(UNSEEN_CLASSES)
	test_id = np.asarray(UNSEEN_CLASSES)
	att_pro = attribute[test_id]      
	attributes = torch.from_numpy(attribute)

###################################################################
	test_features_all =  np.load(FEATURE_DIR+FEATURES_TEST_NAME)
	if NORMALIZE_VIS == True:
		test_features_all = preprocessing.normalize(test_features_all, axis=1, copy=False)	
	test_label_all   =  np.load(FEATURE_DIR+LABELS_TEST_NAME)

	test_features = []
	test_label = []
	test_seen_features = []
	test_seen_label = []
	
	for tindx in range(len(test_features_all)):
		if test_label_all[tindx] in UNSEEN_CLASSES:	
#TODO...verify
			test_features.append(test_features_all[tindx])
			test_label.append(test_label_all[tindx])
		else:			
			test_seen_features.append(test_features_all[tindx])
			test_seen_label.append(test_label_all[tindx])

	test_features = np.asarray(test_features)
	test_label = np.asarray(test_label).astype(int)
	test_seen_features = np.asarray(test_seen_features)
	test_seen_label = np.asarray(test_seen_label).astype(int)
##################################################################

	test_features = torch.from_numpy(test_features)
	test_label=torch.from_numpy(test_label).unsqueeze(1)
	test_seen_features = torch.from_numpy(test_seen_features)
	test_seen_label=torch.from_numpy(test_seen_label)

	test_attributes = torch.from_numpy(att_pro).float()


#TODO
# self.features_array = torch.FloatTensor(self.features_array)
# self.features_array = self.features_array.cuda() if use_cuda else self.features_array
# self.labels_array = torch.LongTensor(self.labels_array)
# self.labels_array = self.labels_array.cuda() if use_cuda else self.labels_array


#TODO:...use squeeze and np.unique
######################
#D train_features =  #(19832L, 2048L) tensor of train seen features
#D train_label =  #(19832L, 1L) tensor of train seen labels
#D attributes = # (50, 85) tensor of all seen/unseen embeddings
#D all_attributes = # (50, 85) NUMPY of all seen/unseen embeddings
######################

######################
# test_features = # (5685L, 2048L) tensor of only unseen features 
# test_label= # (5685L, 1L) tensor of only unseen labels
#D testclasses_id =array([ 6,  8, 22, 23, 29, 30, 33, 40, 46, 49]) # NUMPY of unseen class no.
#D test_attributes = # (10L, 85L) tensor of only unseen attributes
# test_seen_features = # (4958L, 2048L) tensor of ONLY SEEN TEST test features
# test_seen_label = # torch.Size([4958]) 1-D tensor of ONLY SEEN TEST LABELS
#D test_id = #(10) NUMPY ARRAY OF unseen classes in test set
#D test_id = array([ 6,  8, 22, 23, 29, 30, 33, 40, 46, 49])
######################


	train_data = TensorDataset(train_features,train_label)

	print("init networks")
#DTODO: CHNAGE THE NETWORK...........  attribute_network = size of embedding, size of visualfeat/2, size of visualfeat 
#                                     relation_network =  2* size of visualfeat, some hidden size ....evnetaully this networks output is 1
	attribute_network = AttributeNetwork(DIM_LANGAUGE,DIM_STGCN/2,DIM_STGCN)
	relation_network = RelationNetwork(2*DIM_STGCN,DIM_STGCN/2)

	attribute_network.cuda(GPU)
	relation_network.cuda(GPU)

	attribute_network_optim = torch.optim.Adam(attribute_network.parameters(),lr=LEARNING_RATE,weight_decay=1e-5)
	attribute_network_scheduler = StepLR(attribute_network_optim,step_size=200000,gamma=0.5)
	relation_network_optim = torch.optim.Adam(relation_network.parameters(),lr=LEARNING_RATE)
	relation_network_scheduler = StepLR(relation_network_optim,step_size=200000,gamma=0.5)

	print("training...")
	last_accuracy = 0.0

	for episode in range(EPISODE):
		attribute_network_scheduler.step(episode)
		relation_network_scheduler.step(episode)

		train_loader = DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True)

		batch_features,batch_labels = train_loader.__iter__().next()

		sample_labels = []
		for label in batch_labels.numpy():
			if label not in sample_labels:
				sample_labels.append(label)
		# pdb.set_trace()
		
		sample_attributes = torch.Tensor([all_attributes[i] for i in sample_labels]).squeeze(1)
		class_num = sample_attributes.shape[0]

		batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
		sample_features = attribute_network(Variable(sample_attributes).cuda(GPU)) #k*312


		sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_SIZE,1,1)
		batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
		batch_features_ext = torch.transpose(batch_features_ext,0,1)
		
		#print(sample_features_ext)
		#print(batch_features_ext)

#DTODO: CHANGE 4096        
		relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,2*DIM_STGCN)
		# pdb.set_trace()
		relations = relation_network(relation_pairs).view(-1,class_num)
		#print(relations)

		# re-build batch_labels according to sample_labels
		sample_labels = np.array(sample_labels)
		re_batch_labels = []
		for label in batch_labels.numpy():
			index = np.argwhere(sample_labels==label)
			re_batch_labels.append(index[0][0])
		re_batch_labels = torch.LongTensor(re_batch_labels)
		# pdb.set_trace()
		
		# loss
		mse = nn.MSELoss().cuda(GPU)
		one_hot_labels = Variable(torch.zeros(BATCH_SIZE, class_num).scatter_(1, re_batch_labels.view(-1,1), 1)).cuda(GPU)
		loss = mse(relations,one_hot_labels)
		# pdb.set_trace()

		# update
		attribute_network.zero_grad()
		relation_network.zero_grad()

		loss.backward()

		attribute_network_optim.step()
		relation_network_optim.step()

		if (episode+1)%100 == 0:
				print("episode:",episode+1,"loss",loss.data[0])

		if (episode+1)%2000 == 0:
			# test
			print("Testing...")

			def compute_accuracy(test_features,test_label,test_id,test_attributes):
				
				test_data = TensorDataset(test_features,test_label)
				test_batch = 32
				test_loader = DataLoader(test_data,batch_size=test_batch,shuffle=False)
				total_rewards = 0
				# fetch attributes
				# pdb.set_trace()

				sample_labels = test_id
				sample_attributes = test_attributes
				class_num = sample_attributes.shape[0]
				test_size = test_features.shape[0]
				
				print("class num:",class_num)
				
				for batch_features,batch_labels in test_loader:

					batch_size = batch_labels.shape[0]

					batch_features = Variable(batch_features).cuda(GPU).float()  # 32*1024
					sample_features = attribute_network(Variable(sample_attributes).cuda(GPU).float())

					sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1)
					batch_features_ext = batch_features.unsqueeze(0).repeat(class_num,1,1)
					batch_features_ext = torch.transpose(batch_features_ext,0,1)

#DTODO: CHANGE 4096        
					relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,2*DIM_STGCN)
					relations = relation_network(relation_pairs).view(-1,class_num)

					# re-build batch_labels according to sample_labels

					re_batch_labels = []
					for label in batch_labels.numpy():
						index = np.argwhere(sample_labels==label)
						re_batch_labels.append(index[0][0])
					re_batch_labels = torch.LongTensor(re_batch_labels)
					# pdb.set_trace()


					_,predict_labels = torch.max(relations.data,1)
	
					rewards = [1 if predict_labels[j]==re_batch_labels[j] else 0 for j in range(batch_size)]
					total_rewards += np.sum(rewards)
				test_accuracy = total_rewards/1.0/test_size

				return test_accuracy
			
#DTODO: CHANGE np.arrange(50)....to 60...NUMPY ARRAY [0....59]
			zsl_accuracy = compute_accuracy(test_features,test_label,test_id,test_attributes)
			gzsl_unseen_accuracy = compute_accuracy(test_features,test_label,np.arange(NO_CLASSES),attributes)
			gzsl_seen_accuracy = compute_accuracy(test_seen_features,test_seen_label,np.arange(NO_CLASSES),attributes)
			
			H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)
			
			print('zsl:', zsl_accuracy)
			print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))
			

			if zsl_accuracy > last_accuracy:

				# save networks
				torch.save(attribute_network.state_dict(),"./models_LTC/zsl_attribute_network_v1.pkl")
				torch.save(relation_network.state_dict(),"./models_LTC/zsl_relation_network_v1.pkl")

				print("save networks for episode:",episode)

				last_accuracy = zsl_accuracy



if __name__ == '__main__':
	main()
