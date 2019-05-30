# -*- coding: utf-8 -*-
"""
Created on 2019/5/26

@Author: xhj
"""

import os


"""
path and dataset parameters
"""
DATA_PATH = '../../../datasets/'
PASCAL_PATH = os.path.join(DATA_PATH, 'VOCtrainval_06-Nov-2007')
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')
WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')
WEIGHTS_FILE = None


"""
model parameters
"""
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
FLIPPED = True             # flip the image or not
IMAGE_SIZE = 448
CELL_SIZE = 7              # the number of grids in output feature map
BOXES_PER_CELL = 2         # the number of bbox of every grids
ALPHA = 0.1                # the parameter for leaky relu
DISP_CONSALE = False

OBJECT_SCALE = 1.
NOOBJECT_SCALE = .5        # the scale used in loss function for none obj grids
CLASS_SCALE = 2.0          # this scale is not contained in paper(used for class penalizes)
COORD_SCALE = 5.           # the scale used in loss function for obj grids


"""
solver parameters
"""
GPU = ''
# learning_rate = learning_rate * decay_rate^(global_step/decay_step)
LEARNING_RATE = 1e-4
DECAY_STEP = 30000
DECAY_RATE = 0.1
STAIRCASE = True          # if False: global//decay_step
BATCH_SIZE = 128
MAX_ITER = 15000
SUMMARY_ITER = 10         # update the summary file after SUMMARY_ITER's iterations
SAVE_ITER = 1000          # save the weights parameters after SAVE_ITER's iterations


"""
test parameters
"""
THRESHOLD = 0.2           # P < this threshold is not a classes
IOU_THRESHOLD = 0.5       # the threshold used for non-maximum supression