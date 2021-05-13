# 1 AMeFu-Net
Repository for the paper :

**Depth Guided Adaptive Meta-Fusion Network for Few-shot Video Recognition**

[[paper](https://arxiv.org/abs/2010.09982)]

![](https://upload-images.jianshu.io/upload_images/9933353-a0414d86bce9bee5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

If you have any questions, feel free to contact me. My email is yqfu18@fudan.edu.cn.

# 2 setup and datasets
A new virtual environment is suggested.

Before running our code, you are suggested to set up the virtual environment of Monodepth2. We use the Monodepth2 model to extract our depth frames in our paper. 

## 2.1 setup monodepth2
Following [the monodepth2 repo](https://github.com/nianticlabs/monodepth2) to finish the **Setup** and **Prediction for a single image** steps.


## 2.2 preparing the datasets
Totally three datasets are used in our experiments, including Kinetics, UCF101, and HMDB51. 

We suppose that you have downloaded the datasets by yourself.  Sorry for we could not provide the original video datasets. 

There are mainly three steps to processing our dataset.  We also provide the scripts we use for your convenience.

1. extracting the video frames.
`sources/video_jpg.py`
2. calculate the number of frames per video.
`sources/n_frames.py`
3. extracting the depth frames for each video.
`sources/generate_depth.py`

Take the Kinetics dataset as an example, our final datasets are organized as follows: 

```
miniKinetics_frames
├── class1
│   ├── video1
│   │   ├── image_00001.jpg
│   │   ├── image_00002.jpg
│   │   ├── ...
│   │   ├── monodepth
│   │   │   ├── image_00001_disp.jpeg
│   │   │   └── image_00002_disp.jpeg
│   │   │   └── ...
│   │   └── n_frames
│   └── video2
│       ├── image_00001.jpg
│       ├── image_00002.jpg
│       ├── ...
│       ├── monodepth
│       │   ├── image_00001_disp.jpeg
│       │   └── image_00002_disp.jpeg
│       │   └── ...
│       └── n_frames
└── class2
└── ...
```


# 3 network testing

usage for Kinetics dataset:
```
CUDA_VISIBLE_DEVICES=0 python network_test.py --dataset kinetics --ckp ./result/model_full_kinetics.pkl --k_shot 1
```

usage for UCF101  dataset:
```
CUDA_VISIBLE_DEVICES=0 python network_test.py --dataset ucf --ckp ./result/model_full_ucf.pkl --k_shot 1
```

usage for HMDB51 dataset:
```
CUDA_VISIBLE_DEVICES=0 python network_test.py --dataset hmdb --ckp ./result/model_full_hmdb.pkl --k_shot 1
```

The models will be released soon.

Due to the randomness of the few-shot learning,  the result may vary. 

# 4 network training
Take Kinetics as an example:

If ''out of the memory" occurred, then two cards are required.

```
CUDA_VISIBLE_DEVICES=0,1 python network_train_meta_learning.py --dataset kinetics --exp_name test --pre_model_rgb ./result/kinetics_rgb_submodel.pkl --pre_model_depth ./result/kinetics_depth_submodel.pkl
```

The submodels will also be provided.

If you want to train the submodels by yourself,  we refer you to our previous work [emboded few-shot learning](https://github.com/lovelyqian/Embodied-One-Shot-Video-Recognition).


# 5 Citing

Please cite our paper if you find this code useful for your research.

```
@inproceedings{fu2020depth,
  title={Depth Guided Adaptive Meta-Fusion Network for Few-shot Video Recognition},
  author={Fu, Yuqian and Zhang, Li and Wang, Junke and Fu, Yanwei and Jiang, Yu-Gang},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={1142--1151},
  year={2020}
}
```
