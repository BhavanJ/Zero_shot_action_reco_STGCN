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

from sklearn import preprocessing


plt.ion()
use_cuda = torch.cuda.is_available()

#TODO: for loss function u MIGHT WANT TO SET grad function to zero
#TODO: Verify the below thing about unseen classes
UNSEEN_CLASSES = [3,8,11,16,51]
DIM_STGCN = 256
DIM_LANGAUGE = 700
MARGIN = 0.1
N_EPOCH = 200
NO_CLASS = 60
BATCH_SIZE = 64
LR = 0.001
MOMENTUN = 0.9

NORMALIZE_VIS = True
NORMALIZE_EMB = True

LANG_EMB_RANDOM = True


FEATURE_DIR = '/home/bjasani/Desktop/CMU_HW/VLR_project/STGN/st-gcn/work_dir/NTU-RGB-D_zero_shot/xview/ST_GCN_test_2/'
SKELETON_DIR = '/home/bjasani/Desktop/CMU_HW/VLR_project/STGN/st-gcn/data/NTU-RGB-D_zero_shot/xview/'

FEATURES_TRAIN_NAME 	= 'features_array_train_exp_3.npy'
FEATURES_TEST_NAME  	= 'features_array_test_exp_3.npy'
LABELS_TRAIN_NAME       = 'labels_array_train_exp_3.npy'
LABELS_TEST_NAME    	= 'labels_array_test_exp_3.npy'
LANGUAGE_EMBEDDING_NAME = 'bigram_embeddings.npy'


print('############################')
print('############################')
print('############################')
print('############################')
print('LANGUAGE_EMBEDDING_RANDOM: ', LANG_EMB_RANDOM)
print('############################')
print('############################')
print('############################')
print('############################')

langauge_embeddings = np.load(FEATURE_DIR+LANGUAGE_EMBEDDING_NAME)

if LANG_EMB_RANDOM == True:
	langauge_embeddings = np.random.random(langauge_embeddings.shape)


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

			loss_val += torch.sum( torch.max(zero_var, margin_var - torch.sum((true_embedding*vision_feature_projected),dim=1) + torch.sum((langauge_embeddings[cls_no].view(1,-1).repeat(batch_sz,1)*vision_feature_projected),dim=1)  ),dim=0 )
			#pdb.set_trace()	
	
		loss_val = loss_val/batch_sz

		return loss_val

class DataLoaderSTGCN(Dataset):
	def __init__(self,train_mode):

		if train_mode == True:
			self.features_array =  np.load(FEATURE_DIR+FEATURES_TRAIN_NAME)
			if NORMALIZE_VIS == True:
				self.features_array = preprocessing.normalize(self.features_array, axis=1, copy=False)	
			self.labels_array   =  np.load(FEATURE_DIR+LABELS_TRAIN_NAME)

			#norm = np.linalg.norm(self.features_array,ord = 1, axis = 1)
 			#np.sqrt(np.sum((self.features_array[0])*(self.features_array[0])))

		if train_mode == False:			
			self.features_array =  np.load(FEATURE_DIR+FEATURES_TEST_NAME)
			if NORMALIZE_VIS == True:
				self.features_array = preprocessing.normalize(self.features_array, axis=1, copy=False)	
			self.labels_array   =  np.load(FEATURE_DIR+LABELS_TEST_NAME)

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


def test_accuracy():


	correct = 0.0
	correct_top2_all = 0.0
	correct_top3_all = 0.0
	correct_top5_all = 0.0
	total = 0.0

	correct_unseen = 0.0
	correct_unseen_top2 = 0.0
	correct_unseen_top3 = 0.0
	correct_unseen_top5 = 0.0
	total_unseen = 0.0	

	correct_only_unseen = 0.0
	correct_unseen_only_top2 = 0.0
	correct_unseen_only_top3 = 0.0

	for batch_idx,batch_data in enumerate(test_dataloader):

		STGCN_feature = Variable(batch_data['feature']) 
		class_label = Variable(batch_data['label'])
		projected_STGCN_feature = net(STGCN_feature)
		_,predicted_cls = torch.max(torch.sum(langauge_embeddings*projected_STGCN_feature,dim=1),dim=0)

		_,top2_all = torch.topk(torch.sum(langauge_embeddings*projected_STGCN_feature,dim=1), k=2)
		_,top3_all = torch.topk(torch.sum(langauge_embeddings*projected_STGCN_feature,dim=1), k=3)		
		_,top5_all = torch.topk(torch.sum(langauge_embeddings*projected_STGCN_feature,dim=1), k=5)
		
		top2_all = top2_all.data.cpu().numpy()
		top3_all = top3_all.data.cpu().numpy()
		top5_all = top5_all.data.cpu().numpy()

		if class_label.data[0][0] in top2_all:
			correct_top2_all += 1.0
		if class_label.data[0][0] in top3_all:
			correct_top3_all += 1.0
		if class_label.data[0][0] in top5_all:
			correct_top5_all += 1.0

		if predicted_cls.data[0] == class_label.data[0][0]:
			correct+=1.0
		total+=1.0

		
		if class_label.data[0][0] in UNSEEN_CLASSES:
			total_unseen += 1.0
			if class_label.data[0][0] == predicted_cls.data[0]:
				correct_unseen += 1.0
			if class_label.data[0][0] in top2_all:
				correct_unseen_top2 += 1.0
			if class_label.data[0][0] in top3_all:
				correct_unseen_top3 += 1.0
			if class_label.data[0][0] in top5_all:
				correct_unseen_top5 += 1.0


			
			_,argmax_uscls = torch.max(torch.sum(langauge_embeddings*projected_STGCN_feature,dim=1)[UNSEEN_CLASSES],dim=0)
			predicted_unseen_only_cls = UNSEEN_CLASSES[argmax_uscls.data[0]]
			if class_label.data[0][0] == predicted_unseen_only_cls:
				correct_only_unseen	+= 1.0

			_,top_us_only_cls2 = torch.topk(torch.sum(langauge_embeddings*projected_STGCN_feature,dim=1)[UNSEEN_CLASSES],k=2)
			_,top_us_only_cls3 = torch.topk(torch.sum(langauge_embeddings*projected_STGCN_feature,dim=1)[UNSEEN_CLASSES],k=3)
			top_us_only_cls2 = top_us_only_cls2.data.cpu().numpy()	
			top_us_only_cls3 = top_us_only_cls3.data.cpu().numpy()	

			if class_label.data[0][0] == UNSEEN_CLASSES[top_us_only_cls2[0]] or class_label.data[0][0] == UNSEEN_CLASSES[top_us_only_cls2[1]]:
				correct_unseen_only_top2 += 1.0
			if class_label.data[0][0] == UNSEEN_CLASSES[top_us_only_cls3[0]] or class_label.data[0][0] == UNSEEN_CLASSES[top_us_only_cls3[1]] or class_label.data[0][0] == UNSEEN_CLASSES[top_us_only_cls3[2]]:
				correct_unseen_only_top3 += 1.0

			# pdb.set_trace()		
	
	acc = correct/total*100.0
	acc_top2_all = correct_top2_all/total*100.0 
	acc_top3_all = correct_top3_all/total*100.0
	acc_top5_all = correct_top5_all/total*100.0 

	unseen_acc 		= correct_unseen/total_unseen*100.0
	unseen_acc_top2 =  correct_unseen_top2/total_unseen*100.0 
	unseen_acc_top3 = correct_unseen_top3/total_unseen*100.0
	unseen_acc_top5 = correct_unseen_top5/total_unseen*100.0

	unseen_only_acc = correct_only_unseen/total_unseen*100.0
	unseen_only_acc_top2 = correct_unseen_only_top2/total_unseen*100.0
	unseen_only_acc_top3 = correct_unseen_only_top3/total_unseen*100.0

	print('###############')
	print('All videos all classes       : Top1: {} Top2: {} Top3: {} Top5: {}'.format(acc,acc_top2_all,acc_top3_all,acc_top5_all))
	print('Unseen videos all classes    : Top1: {} Top2: {} Top3: {} Top5: {}'.format(unseen_acc,unseen_acc_top2,unseen_acc_top3,unseen_acc_top5))
	print('Unseen videos unseen classes : Top1: {} Top2: {} Top3: {} Top5: - '.format(unseen_only_acc,unseen_only_acc_top2,unseen_only_acc_top3))
	# print('All videos all classes       :' + ' Top1:' + str(acc) +' Top2:' + str(acc_top2_all) +' Top3:' + str(acc_top3_all) +' Top5:' + str(acc_top5_all))
	# print('Unseen videos all classes    :' + ' Top1:' + str(unseen_acc) +' Top2:' + str(unseen_acc_top2) +' Top3:' + str(unseen_acc_top3) +' Top5:' + str(unseen_acc_top5))
	# print('Unseen videos unseen classes :' + ' Top1:' + str(unseen_only_acc) +' Top2:' + str(unseen_only_acc_top2) +' Top3:' + str(unseen_only_acc_top3) +' Top5: - ' )
	print('################\n')

	return acc,unseen_acc,unseen_only_acc	
	
##################################################################################################################

train_dataset = DataLoaderSTGCN(train_mode = True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dataset = DataLoaderSTGCN(train_mode = False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

if NORMALIZE_EMB == True:
	# norm = np.linalg.norm(langauge_embeddings,ord = 1, axis = 1).reshape(60,1)
	# langauge_embeddings = langauge_embeddings/norm	
	langauge_embeddings = preprocessing.normalize(langauge_embeddings, axis=1, copy=False)

langauge_embeddings = Variable(torch.FloatTensor(langauge_embeddings))
langauge_embeddings = langauge_embeddings.cuda() if use_cuda else langauge_embeddings

net = Devise(in_dim = DIM_STGCN,out_dim = DIM_LANGAUGE)
if use_cuda:
	net = net.cuda()
print(net)

criterion = Custom_devise_loss(margin = MARGIN)
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=MOMENTUN)


print('Before training:')
accuracy,unseen_acc,unseen_only_acc = test_accuracy()
#print('Before training: ' + '  Test Accuracy: ' + str(accuracy) + 'Test unseen acc: ' + str(unseen_acc)+'Test unseen only acc:' + str(unseen_only_acc))    


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

		##TODO...need to set the grad_func for loss to true
		loss = criterion(projected_STGCN_feature,langauge_embeddings,class_label)
		loss.backward()
		optimizer.step()
		running_loss += loss.data[0]

	running_loss = running_loss/len(train_dataloader)
	# print('Epoch: ' + str(epoch) + '  Train Loss: ' + str(running_loss) + '  Test Accuracy: ' + str(accuracy) + 'Test unseen acc: ' + str(unseen_acc)+'Test unseen only acc:' + str(unseen_only_acc))    
	print('Epoch: ' + str(epoch) + '  Train Loss: ' + str(running_loss) )#+ '  Test Accuracy: ' + str(accuracy) + 'Test unseen acc: ' + str(unseen_acc)+'Test unseen only acc:' + str(unseen_only_acc))    
	accuracy,unseen_acc,unseen_only_acc = test_accuracy()
	
print('Finished Training')
pdb.set_trace()

torch.save(net.state_dict(), MODEL_PATH)





####################################################################################################

		# for iii in range(len(batch_data)):
			
		# 	if class_label[iii].data[0] in UNSEEN_CLASSES:
		# 		total_unseen += 1.0
		# 		if class_label[iii].data[0] == predicted_cls[iii].data[0]:
		# 			correct_unseen += 1.0
		# 	pdb.set_trace()		

		# pdb.set_trace()


##Loading all the files
# features_array_test = np.load(FEATURE_DIR+'features_array_test.npy')
# features_array_train = np.load(FEATURE_DIR+'features_array_train.npy')
# labels_array_test = np.load(FEATURE_DIR+'labels_array_test.npy')
# labels_array_train = np.load(FEATURE_DIR+'labels_array_train.npy')
#langauge_embeddings = np.load(FEATURE_DIR+'class_embeddings_temp.npy')


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




