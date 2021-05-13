import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression



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
    for i in range(support_y.shape[0]):
        classId = support_y[i]
        video_feature = support_feature[i]
        if classId not in dict.keys():
            dict[classId] = [video_feature]
        else:
            dict[classId].append(video_feature)

    for classId in dict.keys():
        prototype_ids.append(classId)
        prototype_feature = np.array(dict[classId])
        prototype_feature = np.mean(prototype_feature, axis=0)
        prototype_features.append(prototype_feature)
    prototype_features = np.array(prototype_features)
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

    # get distance
    query_features = []
    for i in range(query_y.shape[0]):
        query_feature = query_feature[i]
        # query_feature = np.mean(query_feature, axis=0)
        query_features.append(query_feature)
    query_features = np.array(query_features)

    distance = cdist(query_features, prototype_features,metric='euclidean')

    # get probability
    distance = torch.FloatTensor(distance)
    probability = torch.nn.functional.softmax(-distance)
    

    # caculate accuracy
    label = query_y
    probability = probability.data.cpu().numpy()
    predicted_y = np.argmax(probability, axis=1)

    #return loss, accuracy
    return predicted_y



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
            predicted_y = one_shot_classifier_prototype_lowerdim(data_result)

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
            prototype_ids, prototype_features = generate_prototypes_tensor_lowerdim(data_result)            
            distance_cosine = cosine_similarity(data_result['query_feature'], prototype_features)
            predicted_y = np.argsort(-distance_cosine)
            predicted_y = predicted_y[:,0]
        else:
            print('classifier type error.')
        return predicted_y




if __name__=='__name__':
    myClassifier=classifier()
