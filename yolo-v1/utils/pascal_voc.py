# -*- coding: utf-8 -*-
"""
Created on 2019/5/26

@Author: xhj
"""

import os
import sys
import cv2
import copy
import pickle
import numpy as np 
import xml.etree.ElementTree as ET

sys.path.append('../yolo/')
import config as cfg


class pascal_voc(object):

	def __init__(self, phase, rebuild=False):

		self.devkit_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
		self.data_path = os.path.join(self.devkit_path, 'VOC2007')
		self.cache_path = cfg.CACHE_PATH
		self.batch_size = cfg.BATCH_SIZE
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.classes = cfg.CLASSES
		self.class2label = dict(zip(self.classes, range(len(self.classes))))
		self.flipped = cfg.FLIPPED
		self.phase = phase
		self.rebuild = rebuild
		self.curor = 0
		self.epoch = 1
		self.gt_labels = None
		self.prepare()

	def load_image(self, image_path, flipped=False):
		"""
		load the image from image path which was input
		"""

		img = cv2.imread(image_path, 1)
		img = cv2.resize(img, (self.image_size, self.image_size))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = np.multiply(1/255., img)
		if flipped:
			img = img[:, ::-1, :]
		return img

	def load_xml(self, index):
		"""
		load the image and the label of this image from xml format files
		Args:
			index: the index of xml files
		"""	

		image_path = self.data_path + '/JPEGImages/' + index + '.jpg'
		