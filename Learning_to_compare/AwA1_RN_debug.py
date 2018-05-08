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
import pdb

# '/home/lz/Workspace/ZSL/data/Animals_with_Attributes2',

parser = argparse.ArgumentParser(description="Zero Shot Learning")
parser.add_argument("-b","--batch_size",type = int, default = 32)
parser.add_argument("-e","--episode",type = int, default= 500000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 1e-5)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters

BATCH_SIZE = args.batch_size
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu

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
    # step 1: init dataset
    print("init dataset")
    
    dataroot = './data'
    dataset = 'AwA1_data'
    image_embedding = 'res101' 
    class_embedding = 'original_att'


    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
# matcontent['features'].shape = (2048, 30475)
# matcontent['labels'].shape = (30475, 1) 

    feature = matcontent['features'].T
    label = matcontent['labels'].astype(int).squeeze() - 1
# features and lables here for ALL THE IMAGES


    matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
# matcontent.keys() = ['allclasses_names', '__header__', '__globals__', 
#'train_loc', 'trainval_loc', 'att', 'val_loc', 'test_seen_loc', 
#'__version__', 'test_unseen_loc']

# matcontent['allclasses_names'].shape = (50, 1)
# matcontent['train_loc'].shape = (16864, 1)
# matcontent['trainval_loc'].shape =(19832, 1)
# matcontent['att'].shape = (85, 50) ...............................THESE IS ATTRIBUTE FOR 50 CLASSES
# matcontent['val_loc'].shape = (7926, 1)
# matcontent['test_seen_loc'].shape = (4958, 1)
# matcontent['test_unseen_loc'].shape =(5685, 1)
##### ABOVE DOESN"T CONTAIN ANY IMAGE FEATURES....IT"S PROBABLT JUST LABELS


    # numpy array index starts from 0, matlab starts from 1
    trainval_loc = matcontent['trainval_loc'].squeeze() - 1
    test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
    test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

##########ABOVE IS IMP...THERE ARE 3 KINDS....TRAIN/TEST_UNSEEN/TEST_SEEN


    attribute = matcontent['att'].T 

    x = feature[trainval_loc] # train_features
    train_label = label[trainval_loc].astype(int)  # train_label
    att = attribute[train_label] # train attributes
    x_test = feature[test_unseen_loc]  # test_feature
    test_label = label[test_unseen_loc].astype(int) # test_label
    x_test_seen = feature[test_seen_loc]  #test_seen_feature
    test_label_seen = label[test_seen_loc].astype(int) # test_seen_label
    test_id = np.unique(test_label)   # test_id
    att_pro = attribute[test_id]      # test_attribute

# test_id = array([ 6,  8, 22, 23, 29, 30, 33, 40, 46, 49])
# bj = np.unique(train_label)
# array([ 0,  1,  2,  3,  4,  5,  7,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
#        19, 20, 21, 24, 25, 26, 27, 28, 31, 32, 34, 35, 36, 37, 38, 39, 41,
#        42, 43, 44, 45, 47, 48])
# bj.shape = (40,)
  
# cj = np.unique(test_label_seen)
# cj.shape = (40,)
# cj
# array([ 0,  1,  2,  3,  4,  5,  7,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
#        19, 20, 21, 24, 25, 26, 27, 28, 31, 32, 34, 35, 36, 37, 38, 39, 41,
#        42, 43, 44, 45, 47, 48])

#test_id = 10 unseen classes, bj = 40 classses in train set, cj = same 40 seen classes in test set




    pdb.set_trace()


    # train set
    train_features=torch.from_numpy(x)
    print(train_features.shape)
#(19832L, 2048L)
    train_label=torch.from_numpy(train_label).unsqueeze(1)
    print(train_label.shape)
# (19832L, 1L)
    # attributes
    all_attributes=np.array(attribute)
    print(all_attributes.shape)
# (50, 85)
    attributes = torch.from_numpy(attribute)
    

   

    # test set
    test_features=torch.from_numpy(x_test)
    print(test_features.shape)
# (5685L, 2048L)
    test_label=torch.from_numpy(test_label).unsqueeze(1)
    print(test_label.shape)
# (5685L, 1L)
    testclasses_id = np.array(test_id)
    print(testclasses_id.shape)
# (10,)
    test_attributes = torch.from_numpy(att_pro).float()
    print(test_attributes.shape)
# (10L, 85L)
    test_seen_features = torch.from_numpy(x_test_seen)
    print(test_seen_features.shape)
# (4958L, 2048L)
test_seen_label = torch.from_numpy(test_label_seen)
# torch.Size([4958])


#TODO:...use squeeze and np.unique
######################
# train_features =  #(19832L, 2048L) tensor of train seen features
# train_label =  #(19832L, 1L) tensor of train seen labels
# attributes = # (50, 85) tensor of all seen/unseen embeddings
# all_attributes = # (50, 85) NUMPY of all seen/unseen embeddings
######################

######################
# test_features = # (5685L, 2048L) tensor of only unseen features 
# test_label= # (5685L, 1L) tensor of only unseen labels
# testclasses_id =array([ 6,  8, 22, 23, 29, 30, 33, 40, 46, 49]) # NUMPY of unseen class no.
# test_attributes = # (10L, 85L) tensor of only unseen attributes
# test_seen_features = # (4958L, 2048L) tensor of ONLY SEEN TEST test features
# test_seen_label = # torch.Size([4958]) 1-D tensor of ONLY SEEN TEST LABELS
# test_id = #(10) NUMPY ARRAY OF unseen classes in test set
# test_id = array([ 6,  8, 22, 23, 29, 30, 33, 40, 46, 49])
######################


    train_data = TensorDataset(train_features,train_label)


    # init network
    print("init networks")

#TODO: CHNAGE THE NETWORK...........  attribute_network = size of embedding, size of visualfeat/2, size of visualfeat 
#                                     relation_network =  2* size of visualfeat, some hidden size ....evnetaully this networks output is 1
    attribute_network = AttributeNetwork(85,1024,2048)
    relation_network = RelationNetwork(4096,400)

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

#TODO: CHANGE 4096        
        relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,4096)
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

#TODO: CHANGE 4096        
                    relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,4096)
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
            
# TODO: CHANGE np.arrange(50)....to 60...NUMPY ARRAY [0....59]
            zsl_accuracy = compute_accuracy(test_features,test_label,test_id,test_attributes)
            gzsl_unseen_accuracy = compute_accuracy(test_features,test_label,np.arange(50),attributes)
            gzsl_seen_accuracy = compute_accuracy(test_seen_features,test_seen_label,np.arange(50),attributes)
            
            H = 2 * gzsl_seen_accuracy * gzsl_unseen_accuracy / (gzsl_unseen_accuracy + gzsl_seen_accuracy)
            
            print('zsl:', zsl_accuracy)
            print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (gzsl_seen_accuracy, gzsl_unseen_accuracy, H))
            

            if zsl_accuracy > last_accuracy:

                # save networks
                torch.save(attribute_network.state_dict(),"./models/zsl_awa1_attribute_network_v33.pkl")
                torch.save(relation_network.state_dict(),"./models/zsl_awa1_relation_network_v33.pkl")

                print("save networks for episode:",episode)

                last_accuracy = zsl_accuracy



if __name__ == '__main__':
    main()