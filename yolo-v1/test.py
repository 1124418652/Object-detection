#-*- coding: utf-8 -*-
"""
Created on 2019/5/29

@Author: xhj
"""

import os
import cv2
import argparse
import numpy as np 
import tensorflow as tf 
import yolo.config as cfg
# from yolo.yolo_v1 import YOLO_V1_Net as YOLO 
from utils.timer import Timer

print(cfg.__dict__)