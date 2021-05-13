import numpy as np
import os
import glob
import torch
import argparse



def parse_args(script):
  parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
  parser.add_argument('--dataset', default='kinetics', help='kinetics/ucf/hmdb')
  parser.add_argument('--classifier', default='cosine', help='few shot classifier')
  parser.add_argument('--VIDEO_FRAMES' , default=16, type=int,  help='number of frames') 
  parser.add_argument('--n_way' , default=5, type=int,  help='class num to classify for FSL')
  
  if script == 'train':
    parser.add_argument('--mode', default='train', type=str, help='')
    parser.add_argument('--k_shot', default=1, type=int,  help='number of labeled data in each class, for training stage')
    parser.add_argument('--save_dir'    , default='./result/', type=str, help='')
    parser.add_argument('--exp_name'    , default='tmp', type=str, help='')

    parser.add_argument('--epoch_nums', default = 6, type = int, help='')             
    parser.add_argument('--train_episodes', default = 2000, type = int, help = '')    # 1000 for ucf and hmdb
    parser.add_argument('--lr_1', default = 0.00002, type = float, help = '')         # 0.00001 for ucf and hmdb
    parser.add_argument('--lr_2', default = 0.00002, type = float, help = '')         # 0.00001 for ucf and hmdb
    parser.add_argument('--lr_step_size', default=6, type = int, help ='')            # 3 for ucf and hmdb
    parser.add_argument('--pre_model_rgb', default =  '', type=str, help='the path of the pretrained rgb-submodel')
    parser.add_argument('--pre_model_depth', default = '', type=str, help='the path pf the pretrained depth-submodel')
    parser.add_argument('--pre_model_fusion', default = None, type=str, help='the path of the pretrained whole model')

  elif script == 'test':
    parser.add_argument('--mode', default='test', type=str, help='')
    parser.add_argument('--k_shot', default=1, type=int, help = 'number of labeled data in each class, for testing stage')
    parser.add_argument('--test_episodes', default=10000, type=int, help = 'number of episodes for testing')
    parser.add_argument('--ckp'    , default='', type=str, help='the path of the ckp')
  

  return parser.parse_args()


