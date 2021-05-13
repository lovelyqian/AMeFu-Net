# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import tqdm
import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

from PIL import Image

def main(image_path,model_name='mono_640x192',ext='jpg'):
    """Function to predict for a single image or folder of images
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(image_path):
        # Only testing on a single image
        paths = [image_path]
        output_directory = os.path.dirname(image_path)
    elif os.path.isdir(image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(image_path, '*.{}'.format(ext)))
        output_directory = image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(image_path))
    output_directory = os.path.join(output_directory,'monodepth')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
 
    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
            scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
            #print('scaled_disp:', scaled_disp)
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            #print('colormapped_im:',colormapped_im)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                idx + 1, len(paths), name_dest_im))

    print('-> Done!')



def resize_numpy_1d_depth(video_dir_path):
    video_frames = len(os.listdir(video_dir_path))-2   # one for numpy_file, one for monodepth_dir
    IMG_INIT_H = 256
    for i in range(video_frames):
        s = "%05d" % (i+1)
        image_name =  'image_' + s + '.jpg'
        image_path = os.path.join(video_dir_path,image_name)
        image_depth_name = 'image_' + s + '_disp.npy'
        image_depth_path = os.path.join(os.path.join(video_dir_path,'monodepth'),image_depth_name)

        image_depth_resized_name = 'image_' + s + '_disp_resized.npy'
        image_depth_resized_path = os.path.join(os.path.join(video_dir_path,'monodepth'),image_depth_resized_name)          #BUG！！！all the original file will be replaced!!!!!!!!
        
        image = Image.open(image_path)
        if (image.size[0] < 224):
            image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)

        image_depth = np.load(image_depth_path)
        image_depth_tensor = torch.from_numpy(image_depth)
        original_height, original_width = image.size[1], image.size[0]    
        image_depth_tensor_resized = torch.nn.functional.interpolate(image_depth_tensor, (original_height, original_width), mode="bilinear", align_corners=False)
        image_depth_tensor_resized = image_depth_tensor_resized[0]
        image_depth_resized = image_depth_tensor_resized.numpy()
       
        np.save(image_depth_resized_path, image_depth_resized)

        print("   Processed {:d} of {:d} images - saved prediction to {}".format(
                i + 1, video_frames, image_depth_resized_path))

    print('-> Done!')
 

def remove_numpy_1d_depth(video_dir_path):
    video_frames = len(os.listdir(video_dir_path))-2   # one for numpy_file, one for monodepth_dir
    IMG_INIT_H = 256
    for i in range(video_frames):
        s = "%05d" % (i+1)
        image_depth_resized_name = 'image_' + s + '_disp_resized.npy'
        image_depth_resized_path = os.path.join(os.path.join(video_dir_path,'monodepth'),image_depth_resized_name)          #BUG！！！al
        if (os.path.exists(image_depth_resized_path)):
            os.remove(image_depth_resized_path) 
            print("   Processed {:d} of {:d} images - removed {}".format(
                i + 1, video_frames, image_depth_resized_path))

    print('-> Done!')



def generate_resized_numpy_1d_depth_list(video_dir_path):
    video_frames = len(os.listdir(video_dir_path))-2   # one for numpy_file, one for monodepth_dir
    IMG_INIT_H = 256
    depth_1d_list=[]
    for i in range(video_frames):
        s = "%05d" % (i+1)
        image_name =  'image_' + s + '.jpg'
        image_path = os.path.join(video_dir_path,image_name)
        image_depth_name = 'image_' + s + '_disp.npy'
        image_depth_path = os.path.join(os.path.join(video_dir_path,'monodepth'),image_depth_name)

        image_depth_resized_name = 'image_' + s + '_disp_resized.npy'
        image_depth_resized_path = os.path.join(os.path.join(video_dir_path,'monodepth'),image_depth_resized_name)          #BUG！！！all the original file will be replaced!!!!!!!!

        image = Image.open(image_path)
        if (image.size[0] < 224):
            image = image.resize((224, IMG_INIT_H), Image.ANTIALIAS)

        image_depth = np.load(image_depth_path)
        image_depth_tensor = torch.from_numpy(image_depth)
        original_height, original_width = image.size[1], image.size[0]     #TODO, to check, true
        image_depth_tensor_resized = torch.nn.functional.interpolate(image_depth_tensor, (original_height, original_width), mode="bilinear", align_corners=False)
        image_depth_tensor_resized = image_depth_tensor_resized[0]
        image_depth_resized = image_depth_tensor_resized.numpy()

        #np.save(image_depth_resized_path, image_depth_resized)
        depth_1d_list.append(image_depth_resized)

        print("   Processed {:d} of {:d} images of video {}".format(
                i + 1, video_frames, video_dir_path))
    
    print('-> Done!')
    return depth_1d_list


if __name__ == '__main__':
    # simple test
    # main(image_path='./asset')
    #resize_numpy_1d_depth('./assets')
   
    dataset_dir='/DATACENTER/2/lovelyqian/Kinetics/Kinetics/miniKinetics_frames/'
    dirs = os.listdir(dataset_dir)

    import pickle
    dir_id = 0
    sub_dir_id = 0
    for dir in tqdm.tqdm(dirs):
        dir_id = dir_id +1
        sub_dir_id = 0 
        i_file_name = 'depth_1d_resized_'+ dir + '.pkl'
        aim_depth_1d_pkl_path = os.path.join(dataset_dir, i_file_name)
        aim_depth_1d_pkl_file = open(aim_depth_1d_pkl_path,'wb')
        content = {}

        dir_path = os.path.join(dataset_dir, dir)
        if os.path.isdir(dir_path):
            sub_dirs = os.listdir(dir_path)
            for video_dir in sub_dirs:
                sub_dir_id = sub_dir_id + 1
                video_dir_path = os.path.join(dir_path,video_dir)
                if os.path.isdir(video_dir_path):
                    '''
                    print(video_dir_path)
                    if os.path.exists(os.path.join(video_dir_path,'monodepth')):
                        print(video_dir_path, 'already done.')
                    else:
                        main(video_dir_path)
                    '''
                    #resize_numpy_1d_depth(video_dir_path)
                    #remove_numpy_1d_depth(video_dir_path)
                    content[video_dir_path] = generate_resized_numpy_1d_depth_list(video_dir_path)
                    print("   Processed {:d} sub_video of {:d} video_action".format( sub_dir_id, dir_id ))
    pickle.dump(content, aim_depth_1d_pkl_file)


