import torch

from utils import *


class EpisodeDataloader():
    '''
    get_episode: return episode
    shuffle label every episode
    '''
    def __init__(self, params):
        self.dataset = params.dataset
        self.mode = params.mode
        self.n_way = params.n_way
        self.k_shot = params.k_shot

        self.video_frames = params.VIDEO_FRAMES
        self.frame_dir = FRAME_DIR[self.dataset]
        if(self.mode == 'train'):
            self.dataset_list = TRAIN_LIST[self.dataset]
        elif(self.mode == 'val'):
            self.dataset_list = VAL_LIST[self.dataset]
        elif(self.mode == 'test'):
            self.dataset_list = TEST_LIST[self.dataset]

        self.data = open(self.dataset_list).readlines()


    def get_episode(self):
        '''
        :return: support_x = n_way * k_shot * video, support_y = n_way * k_shot * y,;
        :return: query_x = 1* video , query_y = 1 * y
        '''

        # handle dataset_info
        dict = {}
        for line in self.data:
            line = line.strip('\n')
            class_name = line.split('/')[0]
            if class_name not in dict.keys():
                dict[class_name] = [line]
            else:
                dict[class_name].append(line)

        # sample n-way class_name and shuffle then
        aim_class_names = random.sample(dict.keys(), self.n_way)
        # sample 1 query_name
        aim_query_name = random.sample(aim_class_names,1)[0]

        # sample n-way * k_shot support sets  and 1 quey video
        support_x = []
        support_x_depth = []
        support_y = []
        query_x = []
        query_x_depth = []
        query_y = []

        # for visulization
        support_y_global = []
        query_y_global = []

        for class_name in aim_class_names:
            # get the additional one for query
            if (class_name == aim_query_name):
                aim_video_infos = random.sample(dict[class_name], self.k_shot +1 )
                video_info = aim_video_infos[0]
                video, video_depth = get_video_fusion_from_video_info_rgb_depth_object(video_info, mode= self.mode, video_frames = self.video_frames, frame_dir = self.frame_dir)
                video_class = get_classname_from_video_info(video_info)
                video_y = aim_class_names.index(video_class)
                query_x.append(video)
                query_x_depth.append(video_depth)
                query_y.append(video_y)

                # for visulization
                video_label=get_label_from_video_info(video_info,self.dataset_list)
                query_y_global.append(video_label)

                aim_video_infos=aim_video_infos[1:]
            else:
                aim_video_infos = random.sample(dict[class_name],self.k_shot)

            # sample support set
            for video_info in aim_video_infos:
                video, video_depth = get_video_fusion_from_video_info_rgb_depth_object(video_info, mode= self.mode, video_frames = self.video_frames, frame_dir = self.frame_dir)
                video_class = get_classname_from_video_info(video_info)
                video_y = aim_class_names.index(video_class)
                support_x.append(video)
                support_x_depth.append(video_depth)
                support_y.append(video_y)

                # for visulization
                video_label=get_label_from_video_info(video_info,self.dataset_list)
                support_y_global.append(video_label)
        support_x = torch.stack(support_x)
        support_x = torch.FloatTensor(support_x)
        support_x_depth = torch.stack(support_x_depth)
        support_x_depth = torch.FloatTensor(support_x_depth)
        support_y = torch.FloatTensor(support_y)
        
        query_x = torch.stack(query_x)
        query_x = torch.FloatTensor(query_x)
        query_x_depth = torch.stack(query_x_depth)
        query_x_depth = torch.FloatTensor(query_x_depth)
        query_y = torch.FloatTensor(query_y)

        support_y_global = torch.FloatTensor(support_y_global)
        query_y_global = torch.FloatTensor(query_y_global)
        return ({'support_x': support_x, 'support_y': support_y, 'query_x': query_x, 'query_y': query_y, 'support_x_depth':support_x_depth, 'query_x_depth':query_x_depth, 'support_y_global': support_y_global, 'query_y_global': query_y_global})



    def get_episode_multi_depth(self):
        '''
        :return: support_x = n_way * k_shot * video, support_y = n_way * k_shot * y,;
        :return: query_x = 1* video , query_y = 1 * y
        '''
        # handle dataset_info
        dict = {}
        for line in self.data:
            line = line.strip('\n')
            class_name = line.split('/')[0]
            if class_name not in dict.keys():
                dict[class_name] = [line]
            else:
                dict[class_name].append(line)

        # sample n-way class_name and shuffle then
        aim_class_names = random.sample(dict.keys(), self.n_way)
        # sample 1 query_name
        aim_query_name = random.sample(aim_class_names,1)[0]

        # sample n-way * k_shot support sets  and 1 quey video
        support_x = []
        support_x_depth1 = []
        support_x_depth2 = []
        support_x_depth3 = []
        support_y = []

        query_x = []
        query_x_depth1 = []
        query_x_depth2 = []
        query_x_depth3 = []
        query_y = []

        # for visulization
        support_y_global = []
        query_y_global = []


        for class_name in aim_class_names:
            # get the additional one for query
            if (class_name == aim_query_name):
                aim_video_infos = random.sample(dict[class_name], self.k_shot +1 )
                video_info = aim_video_infos[0]
                video, video_depth = get_video_fusion_from_video_info_rgb_depth_object_multi_depth(video_info, mode= self.mode, video_frames = self.video_frames, frame_dir = self.frame_dir)
                video_class = get_classname_from_video_info(video_info)
                video_y = aim_class_names.index(video_class)
                query_x.append(video)
                query_x_depth1.append(video_depth[0])
                query_x_depth2.append(video_depth[1])
                query_x_depth3.append(video_depth[2])
                query_y.append(video_y)

                # for visulization
                video_label=get_label_from_video_info(video_info,self.dataset_list)
                query_y_global.append(video_label)
                aim_video_infos=aim_video_infos[1:]
            else:
                aim_video_infos = random.sample(dict[class_name], self.k_shot)
            # sample support set
            for video_info in aim_video_infos:
                # support depends on it's mode
                video, video_depth = get_video_fusion_from_video_info_rgb_depth_object_multi_depth(video_info, mode= self.mode, video_frames = self.video_frames, frame_dir = self.frame_dir)
                video_class = get_classname_from_video_info(video_info)
                video_y = aim_class_names.index(video_class)
                support_x.append(video)
                support_x_depth1.append(video_depth[0])
                support_x_depth2.append(video_depth[1])
                support_x_depth3.append(video_depth[2])
                support_y.append(video_y)

                # for visulization
                video_label=get_label_from_video_info(video_info,self.dataset_list)
                support_y_global.append(video_label)


        support_x = torch.stack(support_x)
        support_x = torch.FloatTensor(support_x)
        support_x_depth1 = torch.stack(support_x_depth1)
        support_x_depth1 = torch.FloatTensor(support_x_depth1)
        support_x_depth2 = torch.stack(support_x_depth2)
        support_x_depth2 = torch.FloatTensor(support_x_depth2)
        support_x_depth3 = torch.stack(support_x_depth3)
        support_x_depth3 = torch.FloatTensor(support_x_depth3)
        support_x_depth = [support_x_depth1, support_x_depth2, support_x_depth3]
        support_y = torch.FloatTensor(support_y)

        query_x = torch.stack(query_x)
        query_x = torch.FloatTensor(query_x)
        query_x_depth1 = torch.stack(query_x_depth1)
        query_x_depth1 = torch.FloatTensor(query_x_depth1)
        query_x_depth2 = torch.stack(query_x_depth2)
        query_x_depth2 = torch.FloatTensor(query_x_depth2)
        query_x_depth3 = torch.stack(query_x_depth3)
        query_x_depth3 = torch.FloatTensor(query_x_depth3)
        query_x_depth = [query_x_depth1, query_x_depth2, query_x_depth3]
        query_y = torch.FloatTensor(query_y)

        support_y_global = torch.FloatTensor(support_y_global)
        query_y_global = torch.FloatTensor(query_y_global)
        return ({'support_x': support_x, 'support_y': support_y, 'query_x': query_x, 'query_y': query_y, 'support_x_depth':support_x_depth, 'query_x_depth':query_x_depth, 'support_y_global': support_y_global, 'query_y_global': query_y_global})


