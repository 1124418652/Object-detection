#-*- coding: utf-8 -*-
"""
Created on 2019/5/29

@Author: xhj
"""

import os 
import cv2
import argparse
from yolo.yolo_v1 import YOLO_V1_Net
from utils.pascal_voc import Pascal_voc


def run():
	yolo = YOLO_V1_Net()
	dataset = Pascal_voc('train')
	yolo.init_solver(dataset)
	yolo.train()