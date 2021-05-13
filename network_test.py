
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
from classifier import Classifier
from model_AMeFu import model_AMeFu
from episode_dataloader import EpisodeDataloader


class TestNetwork():
    def __init__(self, params, test_result_txt = 'test.txt'):
        # set params
        self.params = params
        self.test_result_txt =  test_result_txt

        # load model
        self.num_classes = NUM_CLASSES_TRAIN[self.params.dataset]
        #self.mymodel = model_fusion_rgb_depth_object(num_classes=self.num_classes)
        self.mymodel = model_AMeFu(num_classes = self.num_classes)
        self.mymodel.load_state_dict(torch.load(self.params.ckp))
        print('model loaded from:', self.params.ckp)
        self.mymodel.eval()
        self.mymodel.cuda()

        # define episode_dataloader
        self.myEpisodeDataloader = EpisodeDataloader(self.params)

        # define one-shot classifier
        self.myClassifier = Classifier(self.params)


    def generate_epoch_features(self,videos,videos_depth, L2=True):
        video_features=[]
        for i in range(videos.shape[0]):  # (D,h,w,c)
            video =videos[i]
            video_depth = videos_depth[i]

            input = Variable(video).cuda()
            input_depth = Variable(video_depth).cuda()

            x = {}
            x['rgb'], x['depth'] = input, input_depth
            feature, output = self.mymodel(x)
            feature = feature.squeeze(0)
            feature = feature.cpu().detach().numpy()
            video_features.append(feature)
        video_features = np.array(video_features)
        return video_features


    def test_network(self):
        # init file
        self.acc_file = open(self.test_result_txt, "w")
        epoch_nums = self.params.test_episodes

        accs = []
        for epoch in range(epoch_nums):
            data = self.myEpisodeDataloader.get_episode()
            support_x, support_y, query_x, query_y = data['support_x'], data['support_y'], data['query_x'], data['query_y']
            support_x_depth, query_x_depth = data['support_x_depth'], data['query_x_depth']

            support_y = support_y.cpu().detach().numpy()
            query_y = query_y.cpu().detach().numpy()

            # for visulization
            support_y_global, query_y_global = data['support_y_global'], data['query_y_global']
            support_y_global = support_y_global.cpu().detach().numpy()
            query_y_global = query_y_global.cpu().detach().numpy()

            # get support_x features and query_x features
            support_features = self.generate_epoch_features(support_x, support_x_depth, L2=True)
            query_features = self.generate_epoch_features(query_x, query_x_depth, L2=True)

            data_result = {}
            data_result['support_feature'], data_result['support_y'], data_result['query_feature'], data_result['query_y'] = support_features, support_y, query_features, query_y

            # one-shot classifier
            predicted_y = self.myClassifier.predict(data_result)
            acc = np.mean(query_y == predicted_y)

            # show result
            print('epoch:', epoch, 'acc:', acc,'avg_acc:',np.mean(accs), 'query_y_global:', query_y_global, 'pre_y_global:', support_y_global)
            print('epoch:', epoch, 'acc:', acc,'avg_acc:',np.mean(accs), 'query_y_global:', query_y_global, 'pre_y_global:', support_y_global, file=self.acc_file)
            accs.append(acc)
        avg_acc = np.mean(accs)
        print('avg_acc:', avg_acc)
        print('avg_acc:', avg_acc, file=self.acc_file)



if __name__ == '__main__':
    params = parse_args('test')
    print(params)
    print(params.dataset)

   
    acc_path = './result/20210512_'+ params.dataset + '_acc_ours_adain_withshift_'+ str(params.k_shot) + 'shot_' + str(params.test_episodes)  + '.txt'
    myTestNetwork = TestNetwork(params, acc_path)
    myTestNetwork.test_network()

