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

import pdb



FEATURE_DIR = './work_dir/NTU-RGB-D_zero_shot/xview/ST_GCN_test/'
SKELETON_DIR = './data/NTU-RGB-D_zero_shot/xview/'

UNSEEN_CLASSES = [3,8,11,16,51]

DIM_STGCN = 256
DIM_LANGAUGE = 600
MARGIN = 0.1
N_EPOCH = 10
NO_CLASS = 60


features_array_test = np.load(FEATURE_DIR+'features_array_test.npy')
features_array_train = np.load(FEATURE_DIR+'features_array_train.npy')
labels_array_test = np.load(FEATURE_DIR+'labels_array_test.npy')
labels_array_train = np.load(FEATURE_DIR+'labels_array_train.npy')
langauge_embeddings = np.load(FEATURE_DIR+'class_embeddings_temp.npy')


##MAKE THESE UNIT NORM
#features_array_test
#features_array_train

class Devise(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Devise, self).__init__()
        self.fc1 = nn.Linear(in_dim,out_dim)
        
    def forward(self, vision_feature):#,language_feature,mode='train'):
    	x = self.fc1( vision_feature )
        return x

    def loss_devise(self,vision_feature_projected,language_feature,class_label.margin):

    	true_embedding = language_feature[class_label]
    	loss_val = 0
    	for cls_no in range(NO_CLASS):
    		if cls_no == class_label:
    			continue
    		else:
    			loss_val += torch.max(0,margin-true_embedding*vision_feature_projected+language_feature[cls_no]*vision_feature_projected)
    	return loss_val    


net = Devise(in_dim = DIM_STGCN,out_dim = DIM_LANGAUGE)
#criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



for epoch in range(N_EPOCH):  
    running_loss = 0.0
    for idx, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        optimizer.zero_grad()
        outputs = net(inputs)
        
        ##TODO...need to set the grad_func for loss to true
        #loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print('Finished Training')

#define which classes are unseen....confusion???
#load all the features...test and train from STGCN
#load all language embeddings
#define the loss function

DIM_STGCN = 256
DIM_LANGAUGE = 600







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


pdb.set_trace()


