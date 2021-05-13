import os
import cv2
import copy
import torch
import pickle
import random
import numpy as np
import subprocess
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch.autograd import Variable




# GLOBAL variable
FRAME_DIR = dict(ucf = '/DATACENTER/2/lovelyqian/UCF101/UCF-101-frames-2/',
                 hmdb = '/DATACENTER/2/lovelyqian/HMDB-51/hmdb51_frames/',
                 kinetics = '/DATACENTER/2/lovelyqian/miniKinetics_frames/')

TRAIN_LIST = dict(ucf = 'splits/ucf_train.list',
                  hmdb = 'splits/hmdb_train.list',
                  kinetics='splits/kinetics_train.list')

VAL_LIST = dict(ucf = 'splits/ucf_val.list',
                hmdb = 'splits/hmdb_val.list',
                kinetics = 'splits/kinetics_val.list')

TEST_LIST = dict(ucf = 'splits/ucf_test.list',
                 hmdb = 'splits/hmdb_test.list',
                 kinetics = 'splits//kinetics_test.list')


NUM_CLASSES_TRAIN = dict(ucf= 70,
                         hmdb = 32,
                         kinetics = 64)


IMG_INIT_H=256
IMG_crop_size = (224,224)








# gloable functions
def transfer_weights(model_from, model_to):
    wf = copy.deepcopy(model_from.state_dict())
    wt = model_to.state_dict()
    for k in wt.keys() :
        #if (not k in wf)):
        if ((not k in wf) | (k=='fc.weight') | (k=='fc.bias')):
            wf[k] = wt[k]
    model_to.load_state_dict(wf)

# for Video Processing
class ClipRandomCrop(torchvision.transforms.RandomCrop):
  def __init__(self, size):
    self.size = size
    self.i = None
    self.j = None
    self.th = None
    self.tw = None

  def __call__(self, img):
    if self.i is None:
      self.i, self.j, self.th, self.tw = self.get_params(img, output_size=self.size)
      #print('crop:', self.i, self.j, self.th, self.tw)
    return torchvision.transforms.functional.crop(img, self.i, self.j, self.th, self.tw)

class ClipRandomHorizontalFlip(object):
  def __init__(self, ratio=0.5):
    self.is_flip = random.random() < ratio

  def __call__(self, img):
    if self.is_flip:
      return torchvision.transforms.functional.hflip(img)
    else:
      return img

def transforms(mode):
    if (mode=='train'):
        random_crop = ClipRandomCrop(IMG_crop_size)
        flip = ClipRandomHorizontalFlip(ratio=0.5)
        toTensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return torchvision.transforms.Compose([random_crop, flip,toTensor,normalize])
    else:   # mode=='test'
        center_crop = torchvision.transforms.CenterCrop(IMG_crop_size)
        toTensor = torchvision.transforms.ToTensor()
        normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return torchvision.transforms.Compose([center_crop,toTensor,normalize])





def get_video_fusion_from_video_info_rgb_depth_object(video_info, mode,video_frames, frame_dir):
    video_frame_path = os.path.join(frame_dir,video_info)
    video_depth_frame_path = os.path.join(video_frame_path, 'monodepth')
    all_frame_count = len(os.listdir(video_frame_path))-2

    # get image_start_id
    if(all_frame_count -video_frames-1 >1):
        if (mode == 'train'):
            quartile = all_frame_count // 4
            image_start1 = random.randint(1,quartile-3)
            image_start2 = random.randint(quartile+1, 2*quartile-3)
            image_start3 = random.randint(2*quartile+1, 3*quartile-3)
            image_start4 = random.randint(3*quartile+1, all_frame_count-3)
            image_starts = [image_start1, image_start2, image_start3, image_start4]
        elif ((mode == 'test') | (mode=='val')):
            quartile = all_frame_count // 4
            image_start1 = 1
            image_start2 = quartile+1
            image_start3 = 2*quartile+1
            image_start4 = 3*quartile+1
            image_starts = [image_start1, image_start2, image_start3, image_start4]
 
    else:
        quartile = all_frame_count // 4
        image_start1 = 1
        image_start2 = quartile+1
        image_start3 = 2*quartile+1
        image_start4 = 3*quartile+1
        image_starts = [image_start1, image_start2, image_start3, image_start4]
 
    
    myTransform = transforms(mode=mode)

    video=[]
    video_depth=[]
    video_frame_path_list=[]
    for image_id in image_starts:
        for i in range(video_frames//4):
            s = "%05d" % image_id
            image_name = 'image_' + s + '.jpg'
            image_depth_name = 'image_' + s + '_disp.jpeg'
            image_path = os.path.join(video_frame_path, image_name)
            image_depth_path = os.path.join(video_depth_frame_path, image_depth_name)
            image = Image.open(image_path)
            image_depth = Image.open(image_depth_path)
            video_init_shape = image.size
            if (image.size[0] < 224):
                image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)
                image_depth = image_depth.resize((224,IMG_INIT_H), Image.ANTIALIAS)

            # to apply the same transform for RGB and depthi, shoule check random_crop in the same position [true]
            image = myTransform(image)
            image_depth = myTransform(image_depth)

            video.append(image)
            video_depth.append(image_depth)
            video_frame_path_list.append(image_id)
        
            image_id += 1
            if (image_id > all_frame_count):
                break
   
    l = len(video)
    if l < video_frames:
        for i in range(video_frames - l):
            video.append(video[l-1])
            video_depth.append(video_depth[l-1])
            video_frame_path_list.append(video_frame_path_list[l-1])
     
    video=torch.stack(video,0)
    video_depth = torch.stack(video_depth,0) 

    return video, video_depth




def get_video_fusion_from_video_info_rgb_depth_object_multi_depth(video_info,mode,video_frames, frame_dir):
    video_frame_path = os.path.join(frame_dir,video_info)
    video_depth_frame_path = os.path.join(video_frame_path, 'monodepth')
    all_frame_count = len(os.listdir(video_frame_path))-2

    # get image_start_id
    if(all_frame_count -video_frames-1 >1):
        if (mode == 'train'):
            quartile = all_frame_count // 4
            image_start1 = random.randint(1,quartile-3)
            image_start11 = random.randint(1,quartile-3)
            image_start12 = random.randint(1,quartile-3)
            image_start13 = random.randint(1,quartile-3)
            image_start2 = random.randint(quartile+1, 2*quartile-3)
            image_start21 = random.randint(quartile+1, 2*quartile-3)
            image_start22 = random.randint(quartile+1, 2*quartile-3)
            image_start23 = random.randint(quartile+1, 2*quartile-3)
            image_start3 = random.randint(2*quartile+1, 3*quartile-3)
            image_start31 = random.randint(2*quartile+1, 3*quartile-3)
            image_start32 = random.randint(2*quartile+1, 3*quartile-3)
            image_start33 = random.randint(2*quartile+1, 3*quartile-3)
            image_start4 = random.randint(3*quartile+1, all_frame_count-3)
            image_start41 = random.randint(3*quartile+1, all_frame_count-3)
            image_start42 = random.randint(3*quartile+1, all_frame_count-3)
            image_start43 = random.randint(3*quartile+1, all_frame_count-3)
            image_starts = [image_start1, image_start2, image_start3, image_start4]
            image_starts1 = [image_start1, image_start2, image_start3, image_start4]
            image_starts2 = [image_start12, image_start22, image_start32, image_start42]
            image_starts3 = [image_start13, image_start23, image_start33, image_start43]
        # get middle 32-frame clip
        elif ((mode == 'test') | (mode=='val')):
            quartile = all_frame_count // 4
            image_start1 = 1
            image_start2 = quartile+1
            image_start3 = 2*quartile+1
            image_start4 = 3*quartile+1
            image_starts = [image_start1, image_start2, image_start3, image_start4]
            image_starts1 = [image_start1, image_start2, image_start3, image_start4]
            image_starts2 = [image_start1, image_start2, image_start3, image_start4]
            image_starts3 = [image_start1, image_start2, image_start3, image_start4]

    else:
        quartile = all_frame_count // 4
        image_start1 = 1
        image_start2 = quartile+1
        image_start3 = 2*quartile+1
        image_start4 = 3*quartile+1

        image_starts = [image_start1, image_start2, image_start3, image_start4]
        image_starts1 = [image_start1, image_start2, image_start3, image_start4]
        image_starts2 = [image_start1, image_start2, image_start3, image_start4]
        image_starts3 = [image_start1, image_start2, image_start3, image_start4]


    myTransform = transforms(mode=mode)

    video=[]
    video_depth1=[]
    video_depth2=[]
    video_depth3=[]
    video_frame_path_list=[]
    for image_id in image_starts:
        for i in range(video_frames//4):
            s = "%05d" % image_id
            image_name = 'image_' + s + '.jpg'
            #image_depth_name = 'image_' + s + '_disp.jpeg'
            image_path = os.path.join(video_frame_path, image_name)
            #image_depth_path = os.path.join(video_depth_frame_path, image_depth_name)
            image = Image.open(image_path)
            #image_depth = Image.open(image_depth_path)
            video_init_shape = image.size
            if (image.size[0] < 224):
                image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)
                #image_depth = image_depth.resize((224,IMG_INIT_H), Image.ANTIALIAS)

            # to apply the same transform for RGB and depthi, shoule check random_crop in the same position [true]
            image = myTransform(image)
            #image_depth = myTransform(image_depth)

            video.append(image)
            #video_depth.append(image_depth)
            video_frame_path_list.append(image_id)

            image_id += 1
            if (image_id > all_frame_count):
                break

    for image_id in image_starts1:
        for i in range(video_frames//4):
            s = "%05d" % image_id
            image_depth_name = 'image_' + s + '_disp.jpeg'
            image_depth_path = os.path.join(video_depth_frame_path, image_depth_name)
            image_depth = Image.open(image_depth_path)
            video_init_shape = image.size
            if (image.size(0) < 224):
                image_depth = image_depth.resize((224, IMG_INIT_H), Image.ANTIALIAS)
            image_depth = myTransform(image_depth)
            video_depth1.append(image_depth)
            image_id += 1
            if (image_id > all_frame_count):
                break


    for image_id in image_starts2:
        for i in range(video_frames//4):
            s = "%05d" % image_id
            image_depth_name = 'image_' + s + '_disp.jpeg'
            image_depth_path = os.path.join(video_depth_frame_path, image_depth_name)
            image_depth = Image.open(image_depth_path)
            video_init_shape = image.size
            if (image.size(0) < 224):
                image_depth = image_depth.resize((224, IMG_INIT_H), Image.ANTIALIAS)
            image_depth = myTransform(image_depth)
            video_depth2.append(image_depth)
            image_id += 1
            if (image_id > all_frame_count):
                break

    for image_id in image_starts3:
        for i in range(video_frames//4):
            s = "%05d" % image_id
            image_depth_name = 'image_' + s + '_disp.jpeg'
            image_depth_path = os.path.join(video_depth_frame_path, image_depth_name)
            image_depth = Image.open(image_depth_path)
            video_init_shape = image.size
            if (image.size(0) < 224):
                image_depth = image_depth.resize((224, IMG_INIT_H), Image.ANTIALIAS)
            image_depth = myTransform(image_depth)
            video_depth3.append(image_depth)
            image_id += 1
            if (image_id > all_frame_count):
                break


    l = len(video)
    if l < video_frames:
        for i in range(video_frames - l):
            video.append(video[l-1])
            video_frame_path_list.append(video_frame_path_list[l-1])

    l1 = len(video_depth1)
    if l1 < video_frames:
        for j in range(video_frames - l1):
            video_depth1.append(video_depth1[l-1])

    l2 = len(video_depth2)
    if l2 < video_frames:
        for j in range(video_frames - l2):
            video_depth2.append(video_depth2[l-1])

    l3 = len(video_depth3)
    if l3 < video_frames:
        for j in range(video_frames - l3):
            video_depth3.append(video_depth3[l-1])

    video=torch.stack(video,0)
    video_depth1 = torch.stack(video_depth1,0)
    video_depth2 = torch.stack(video_depth2,0)
    video_depth3 = torch.stack(video_depth3,0)
    video_depth = [video_depth1, video_depth2, video_depth3]

    return video, video_depth



def get_classname_from_video_info(video_info):
    video_info_splits = video_info.split('/')
    class_num = video_info_splits[0]
    return class_num


def get_classInd(info_list):
    info_list=open(info_list).readlines()
    classlabel=0
    classInd={}
    for info_line in info_list:
        info_line=info_line.strip('\n')
        videoname= get_classname_from_video_info(info_line)
        if videoname not in classInd.keys():
            classInd[videoname]=classlabel
            classlabel = classlabel +1
        else:
            pass
    return classInd


def get_label_from_video_info(video_info,info_list = TRAIN_LIST):
   classname = get_classname_from_video_info(video_info)
   classInd = get_classInd(info_list)
   label = classInd[classname]
   return label




if __name__=='__main__':
    video_info = 'air drumming/-VtLx-mcPds_000012_000022'


    video = get_video_from_video_info(video_info,mode='train',modality='depth-4d')
    print(video.shape)

    label = get_label_from_video_info(video_info)
    

    video, video_depth, video_object_features =  get_video_fusion_from_video_info_rgb_depth_object(video_info, mode='train')
    '''
    from baseline_resnet_pretrained import trainTest
    trainTest()
    '''
    
    print('video frames:',VIDEO_FRAMES)   
    test_info = open(TEST_LIST).readlines()
    for i in range(len(test_info)):
        test_list = test_info[i]
        test_list = test_list.strip('\n')
        video,video_frame_num = get_video_from_video_info_3(test_list,mode='test')
        print(i,video.shape,video.shape[0]==VIDEO_FRAMES,video_frame_num)
