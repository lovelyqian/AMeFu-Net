import cv2
import os
import  numpy as np
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


from options import *
from utils import *
from model_AMeFu import model_AMeFu
from episode_dataloader import EpisodeDataloader
from classifier_with_loss import Classifier



class TrainNetwork():
    def __init__(self, params):
        # get params
        self.params = params

        self.dataset = params.dataset
        self.exp_name = params.exp_name
        self.ckp_path = params.save_dir + self.exp_name
        self.loss_path = self.ckp_path + '/' + 'loss.txt'
        self.epoch_nums = params.epoch_nums
        self.train_episodes = params.train_episodes
        self.lr_1 = params.lr_1
        self.lr_2 = params.lr_2
        self.lr_step_size = params.lr_step_size
        self.num_classes = NUM_CLASSES_TRAIN[self.dataset]
        self.classifier = params.classifier
        self.pre_model_rgb = params.pre_model_rgb
        self.pre_model_depth = params.pre_model_depth
        self.pre_model_fusion = params.pre_model_fusion

        # mkdir ckg_path
        if not os.path.exists(self.ckp_path):
            os.makedirs(self.ckp_path)

        # load model
        self.mymodel = model_AMeFu(num_classes = self.num_classes)
        self.mymodel.submodel_rgb.eval()
        self.mymodel.submodel_depth.eval()
        self.mymodel.instance_norm.train()
        self.mymodel.fc.train()

        # load pretrained model
        if(self.pre_model_rgb):
            self.mymodel.submodel_rgb.load_state_dict(torch.load(self.pre_model_rgb))
            print('submodel_rgb loaded from {}'.format(self.pre_model_rgb))
        if(self.pre_model_depth):
            self.mymodel.submodel_depth.load_state_dict(torch.load(self.pre_model_depth))
            print('submodel_depth loaded from {}'.format(self.pre_model_depth))
        if(self.pre_model_fusion):
            self.mymodel.load_state_dict(torch.load(self.pre_model_fusion))
            print('model loaded from {}'.format(self.pre_model_fusion))



        self.mymodel = torch.nn.DataParallel(self.mymodel)
        self.mymodel.cuda()

        # define episode_dataloader
        self.myEpisodeDataloader = EpisodeDataloader(params)
 
        # define classifier to calculate loss
        self.myClassifier = Classifier(params)


    def finetune_model(self):
        file = open(self.loss_path,'w')

        # define params
        params_list =  list(self.mymodel.module.instance_norm.parameters())
        optimizer_1 = optim.SGD(params_list, lr= self.lr_1, momentum=0.9)
        optimizer_2 = optim.SGD(self.mymodel.module.fc.parameters(), lr= self.lr_2, momentum=0.9)
        scheduler_1 = optim.lr_scheduler.StepLR(optimizer_1, step_size = self.lr_step_size, gamma=0.1)
        scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size = self.lr_step_size, gamma=0.1)
        
        #criterion = nn.CrossEntropyLoss()
        for epoch in range(self.epoch_nums):
            running_loss = 0.0
            accs=[]
            for i_batch in range(self.train_episodes):
                # get the input
                data = self.myEpisodeDataloader.get_episode_multi_depth()
                support_x, support_y, query_x, query_y = data['support_x'], data['support_y'], data['query_x'], data['query_y']
                support_x_depth, query_x_depth = data['support_x_depth'], data['query_x_depth']
    
                # for visulization
                support_y_global, query_y_global = data['support_y_global'], data['query_y_global']
                support_y_global = support_y_global.cpu().detach().numpy()
                query_y_global = query_y_global.cpu().detach().numpy()
                
                # forward
                # warp them as variable
                support_x = Variable(support_x).cuda()
                query_x = Variable(query_x).cuda()
                support_y = Variable(support_y).cuda()
                query_y = Variable(query_y).cuda()
                support_x_shape,query_x_shape = support_x.size(), query_x.size()
                support_x = support_x.view(-1,support_x_shape[2],support_x_shape[3],support_x_shape[4])
                query_x = query_x.view(-1,query_x_shape[2],query_x_shape[3],query_x_shape[4])
            
                support_x.requires_grad_()
                query_x.requires_grad_()

                for j in range(3):
                    # zero_grad
                    optimizer_1.zero_grad()
                    optimizer_2.zero_grad()

                    support_x_depth_single = Variable(support_x_depth[j]).cuda()
                    query_x_depth_single = Variable(query_x_depth[j]).cuda()

                    support_x_depth_single2 = support_x_depth_single.view(-1,support_x_shape[2],support_x_shape[3],support_x_shape[4])
                    query_x_depth_single2 = query_x_depth_single.view(-1,query_x_shape[2],query_x_shape[3],query_x_shape[4])
                
                    support_x_depth_single2.requires_grad_()
                    query_x_depth_single2.requires_grad_()

                    # stack support and query to make the batch_size = 6 (can be divided to 2 or 3 devices)
                    support_query_input = {}
                    support_query_input['rgb'] = torch.cat((support_x, query_x),0)
                    support_query_input['depth'] = torch.cat((support_x_depth_single2, query_x_depth_single2),0)
    
                    support_query_feature, out = self.mymodel(support_query_input)
                    support_features = support_query_feature[0:5,:]
                    query_features = support_query_feature[5,:].unsqueeze(0)

                    data_result = {}
                    data_result['support_feature'], data_result['support_y'], data_result['query_feature'], data_result['query_y'] = support_features, support_y, query_features, query_y

                    # one-shot classifier
                    predicted_y, loss = self.myClassifier.predict(data_result)
                    acc = np.mean(query_y.detach().cpu().numpy() == predicted_y)
                    accs.append(acc)

                    # backward loss
                    loss.backward()
                    optimizer_1.step()
                    optimizer_2.step()
                    running_loss = running_loss+loss.item()



                
                if i_batch % 50 == 49:
                    accuracy = np.mean(accs)
                    print('[%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i_batch + 1, running_loss / 150, accuracy))
                    print ('[%d, %5d] loss: %.3f accuracy: %.3f' %(epoch + 1, i_batch + 1, running_loss / 150, accuracy),file=file)
                    running_loss = 0.0
                    accs=[]
            
            scheduler_1.step()
            scheduler_2.step()
            save_model_path= self.ckp_path +'/model'+str(epoch+1)+'.pkl'
            torch.save(self.mymodel.module.state_dict(),save_model_path)



if __name__ == '__main__':
    params = parse_args('train')
    print(params)
    print(params.dataset)

    myTrainNetwork = TrainNetwork(params)
    myTrainNetwork.finetune_model()

    

