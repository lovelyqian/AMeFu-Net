import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression


def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist(x,y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    cosine_sim_list = []
    for i in range(m):
        y_tmp = y[i].unsqueeze(0)
        x_tmp = x[0].unsqueeze(0)
        #print(x_tmp.size(),y_tmp.size())
        cosine_sim = torch.nn.functional.cosine_similarity(x_tmp,y_tmp)
        cosine_sim_list.append(cosine_sim[0])
    return torch.stack(cosine_sim_list)

def generate_prototypes_tensor_lowerdim(data):
    '''
    data:  dict{'support_feature'[],'support_y'[],'query_feature'[],'query_y'[]}
    return: prototype_ids, prototype_features
    '''
    support_feature, support_y, query_x, query_y = data['support_feature'], data['support_y'], data['query_feature'], data['query_y']
    # get prototype ids and prototype_features
    prototype_ids = []
    prototype_features = []

    dict = {}
    for i in range(support_y.size()[0]):
        classId = support_y[i]
        video_feature = support_feature[i]
        if classId not in dict.keys():
            dict[classId] = [video_feature]
        else:
            dict[classId].append(video_feature)

    for classId in dict.keys():
        prototype_ids.append(classId)
        prototype_feature = torch.stack(dict[classId])
        prototype_feature = torch.mean(prototype_feature, axis=0)
        prototype_features.append(prototype_feature)
    prototype_features = torch.stack(prototype_features)
    return (prototype_ids, prototype_features)


def one_shot_classifier_prototype_lowerdim(data):
    '''
    data: dict{'support_feature[],'support_y'[],'query_feature'[],'query_y'[]}
    return : predicted_y
    '''
    # get input
    support_feature, support_y, query_feature, query_y = data['support_feature'], data['support_y'], data['query_feature'], data['query_y']

    # get prototypes_ids and prototype_features
    prototype_ids, prototype_features = generate_prototypes_tensor_lowerdim(data)
    # print(prototype_ids,prototype_features.shape)

    # get distance
    query_features = []
    for i in range(query_y.size()[0]):
        query_feature = query_feature[i]
        # query_feature = np.mean(query_feature, axis=0)
        query_features.append(query_feature)
    query_features = torch.stack(query_features)
  

    distance = euclidean_dist(query_features, prototype_features)
    probability = torch.nn.functional.softmax(-distance)
    
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()
    label= query_y.long()
    loss = criterion(probability, label) 

    probability = probability.data.cpu().numpy()
    predicted_y = np.argmax(probability, axis=1)

    return predicted_y, loss


class Classifier():
    def __init__(self, params):
        self.classifier = params.classifier
        self.k_shot = params.k_shot

    def predict(self, data_result):
        # data_result: dict
        # classifier_type:
        # 1. protoNet
        # 2. SVM
        # 3. KNN
        # 4. logistic regression
        # 5. classifier NN
        if (self.classifier == 'protonet'):
            predicted_y,loss = one_shot_classifier_prototype_lowerdim(data_result)
        elif (self.classifier == 'SVM'):
            classifier_SVM = SVC(C=10) 
            classifier_SVM.fit(data_result['support_feature'], data_result['support_y'])
            predicted_y = classifier_SVM.predict(data_result['query_feature'])
        elif (self.classifier == 'LR'):
            classifier_LR = LogisticRegression()
            classifier_LR.fit(data_result['support_feature'], data_result['support_y'])
            predicted_y = classifier_LR.predict(data_result['query_feature']) 
        elif (self.classifier == 'KNN'):
            classifier_KNN = KNeighborsClassifier(n_neighbors= self.k_shot)
            classifier_KNN.fit(data_result['support_feature'],data_result['support_y'])
            predicted_y = classifier_KNN.predict(data_result['query_feature'])
        elif(self.classifier == 'cosine'):
            distance_cosine_tensor = cosine_dist(data_result['query_feature'],data_result['support_feature'])
            #print('tensor cosine dist:', distance_cosine_tensor.size())
            #distance_cosine = cosine_similarity(data_result['query_feature'].detach().cpu().numpy(),data_result['support_feature'].detach().cpu().numpy())
            #print('numpyt cosine dist:', distance_cosine)
            probability = torch.nn.functional.softmax(distance_cosine_tensor, dim=-1)
            #print('tensor cosine dist:', distance_cosine_tensor)
            query_y = data_result['query_y']
            import torch.nn as nn
            criterion = nn.CrossEntropyLoss()
            probability = probability.unsqueeze(0)
            label= query_y.long()
            #print('pro:', probability, 'label:',label)
            loss = criterion(probability, label)

            probability = probability.data.cpu().numpy()
            predicted_y = np.argmax(probability, axis=1)


            #predicted_y = np.argsort(-distance_cosine)
            #predicted_y = predicted_y[:,0]
        else:
            print('classifier type error.')
        return predicted_y,loss




if __name__=='__name__':
    myClassifier=classifier()
