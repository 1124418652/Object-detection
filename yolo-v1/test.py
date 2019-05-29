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
from yolo.yolo_v1 import YOLO_V1_Net as YOLO 
from utils.timer import Timer


class Detector(object):

	def __init__(self, net, weight_file):
		"""
		初始化Yolo网络的检测器
		Args:
			net: YOLO_V1_NET 对象
			weight_file: 权重文件的路径
		"""

		self.net = net
		self.weight_file = weight_file

		self.classes = cfg.CLASSES
		self.num_class = len(self.classes)
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.boxes_per_cell = cfg.BOXES_PER_CELL
		self.threshold = cfg.THRESHOLD
		self.iou_threshold = cfg.IOU_THRESHOLD
		self.


if __name__ == '__main__':
	yolo = YOLO()
	detector = Detector(yolo, '')