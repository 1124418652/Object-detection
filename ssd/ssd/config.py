# -*- coding: utf-8 -*-
"""
Created on 2019/6/4

@Author: xhj
"""

import os


"""
model parameters
"""
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
NUM_CLASSES = len(CLASSES) + 1    # 加上一个背景类，在YOLO-v1中不需要加上背景类
IMAGE_SIZE = 300
FEATURE_LAYERS = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
FEATURE_MAP_SIZE = [(38, 38), (19, 19), (10, 10), (5, 5), 
					(3, 3), (1, 1)]    # 提取的 feature map 的尺寸
BOXES_NUM = [4, 6, 6, 6, 4, 4]         # 每个提取出的feature map中每个像素点对应的锚框数目
N_BOXES = [5776, 2166, 600, 150, 36, 4]# 总共8732个先验框
# 表示是否需要L2归一化，只有第四个卷积层的特征图需要
IS_L2NORM = [True, False, False, False, False, False]


"""
parameters of anchor generate
"""
ANCHOR_SIZES = [[21., 45.],       # [min_size, max_size]
				[45., 99.], 
				[99., 153.],
				[153., 207.],
				[207., 261.], 
				[261., 315.]]
ANCHOR_RATIOS = [[2, .5], 
				 [2, .5, 3, 1. / 3], 
				 [2, .5, 3, 1. / 3],
				 [2, .5, 3, 1. / 3], 
				 [2, .5], 
				 [2, .5]]
# step 表示的是各特征图与原图之间的比例关系
ANCHOR_STEPS = [8, 16, 32, 64, 100, 300]
PRIOR_SCALING = [0.1, 0.1, 0.2, 0.2]        # 先验框缩放比例


"""
path and dataset parameters
"""
DATA_PATH = '../../../datasets/'
PASCAL_PATH = os.path.join(DATA_PATH, 'VOCtrainval_06-Nov-2007')
CACHE_PATH = os.path.join('..', 'cache')
WEIGHTS_DIR = '../weight/'
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, 'ssd_vgg_300_weights.ckpt')


"""
data prepare parameters
"""
FLIPPED = True