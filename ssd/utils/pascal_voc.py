# -*- coding: utf-8 -*-
"""
Created on 2019/6/13

@Author: xhj
"""

import os
import sys
import cv2
import copy
import pickle
import numpy as np
import xml.etree.ElementTree as ET

sys.path.append('../ssd/')
import config as cfg 


class Pascal_voc(object):

	def __init__(self, phase, 
				 batch_size=10,
				 rebuild=False, 
				 pascal_path=None,
				 name='pascal'):
		"""
		initialize the object
		Args:
			phase: str type, indicates the phase that dataset was used,
				   {'train', 'validation'}
			rebuild: boolean type, if True, the cache file of labels will
					 be rewrited
			name: str type, the name of dataset, used for naming cache file
		"""

		if not pascal_path:
			pascal_path = cfg.PASCAL_PATH
		if not os.path.exists(pascal_path):
			raise ValueError("Can't find pascal path: %s"%pascal_path)
		self.pascal_path = pascal_path
		self.data_path = os.path.join(self.pascal_path, 'VOCdevkit', 'VOC2007')
		self.name = name
		self.image_size = cfg.IMAGE_SIZE
		self.batch_size = batch_size
		self.classes = cfg.CLASSES
		self.num_classes = cfg.NUM_CLASSES
		self.class2label = dict(zip(self.classes, range(1, self.num_classes)))  # 第0类为背景
		self.flipped = cfg.FLIPPED       # 是否需要左右反向
		self.phase = phase               # 读取训练数据还是测试数据
		self.rebuild = rebuild           # 是否需要重建缓存文件中的数据
		self.cache_path = cfg.CACHE_PATH
		self.cursor = 0                  # 当前读取记录的位置，<= 数据集的大小-1
		self.epoch = 1                   # 完整读取全数据集的次数
		self.gt_labels = None
		self._prepare()


	def _load_image(self, image_path, flipped=False):
		try:
			img = cv2.imread(image_path)
			img = cv2.resize(img, (self.image_size, self.image_size))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		except:
			print("Can't find image: %s"%image_path)
			return None
		means = np.array([123., 117., 104.])
		img = img - means
		if flipped:
			img = img[:, ::-1, :]
		return img


	def _load_xml(self, xml_path):
		"""
		从图片对应的xml文件中提取该图片的标注信息，包含classes以及coordinations
		"""
		try:
			tree = ET.parse(xml_path)
		except:
			print("Can't find xml file: %s"%xml_path)
			return None
		
		shape = tree.find('size')
		width = float(shape.find('width').text)
		height = float(shape.find('height').text)
		w_scale = self.image_size / width
		h_scale = self.image_size / height

		classes = []
		boxes = []
		objs = tree.findall('object')
		for obj in objs:
			box = obj.find('bndbox')
			x1 = min(max(float(box.find('xmin').text)*w_scale, 0), 
				self.image_size - 1) / self.image_size
			y1 = min(max(float(box.find('ymin').text)*h_scale, 0), 
				self.image_size - 1) / self.image_size
			x2 = min(max(float(box.find('xmax').text)*w_scale, 0), 
				self.image_size - 1) / self.image_size
			y2 = min(max(float(box.find('ymax').text)*h_scale, 0), 
				self.image_size - 1) / self.image_size

			box = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
			cls_id = self.class2label[obj.find('name').text.lower().strip()]
			boxes.append(box)
			classes.append(cls_id)

		return np.array(classes), np.array(boxes), len(objs)


	def _load_labels(self):
		"""
		load the labels of all images in dataset
		"""
		cache_file = os.path.join(self.cache_path, \
			'pascal_' + self.phase + '_' + self.name + '_gt_ssd_labels.pkl')
		if os.path.isfile(cache_file) and not self.rebuild:   # 缓存文件存在且不需要重建
			print("Loading gt_labels from: ", cache_file)
			with open(cache_file, 'rb') as fr:
				gt_labels = pickle.load(fr)
				return gt_labels

		print("Loading gt_labels from: ", self.data_path)  # else
		if not os.path.exists(cache_file):
			if not os.path.isdir(self.cache_path):
				os.makedirs(self.cache_path)
		if 'train' == self.phase:
			txt_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
		else:
			txt_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')

		with open(txt_name, 'r') as fr:
			image_indexes = [x.strip() for x in fr.readlines()]

		gt_labels = []
		for index in image_indexes:
			xml_path = os.path.join(self.data_path, 'Annotations', index + '.xml')
			classes, boxes, num = self._load_xml(xml_path)
			if 0 == num:        # 在训练时直接跳过目标数为0的图片
				continue
			image_path = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
			gt_labels.append({'image_path': image_path,
							  'classes': classes,
							  'boxes': boxes,
							  'flipped': False})    # 原始数据，flipped为False

		print("Saving gt_labels to: " + cache_file)
		with open(cache_file, 'wb') as fw:
			pickle.dump(gt_labels, fw)

		return gt_labels


	def _prepare(self):
		"""
		按照要求从文件中读取标签数据到内存中
		"""
		gt_labels = self._load_labels()
		
		if self.flipped:          # 是否需要添加反向数据
			print("Appending horizontanlly-flipped training examples...")
			gt_labels_cp = copy.deepcopy(gt_labels)
			for index in range(len(gt_labels_cp)):
				gt_labels_cp[index]['flipped'] = True
				gt_labels_cp[index]['boxes'][:, 0] = (self.image_size - 1) / self.image_size \
					- gt_labels_cp[index]['boxes'][:, 0]
			gt_labels.extend(gt_labels_cp)

		np.random.shuffle(gt_labels)
		self.gt_labels = gt_labels


	def get_data(self):
		"""
		读取一个batch的数据到内存中，每调用一次该函数，从硬盘中读取一个batch的图片
		"""
		count = 0
		batch_classes = []
		batch_boxes = []
		batch_images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
		
		while count < self.batch_size:
			image_path = self.gt_labels[self.cursor]['image_path']
			flipped = self.gt_labels[self.cursor]['flipped']
			batch_images[count, ...] = self._load_image(image_path, flipped)
			batch_classes.append(self.gt_labels[self.cursor]['classes'])
			batch_boxes.append(self.gt_labels[self.cursor]['boxes'])
			count += 1 
			self.cursor += 1
			if self.cursor >= len(self.gt_labels):
				np.random.shuffle(self.gt_labels)
				self.cursor = 0 
				self.epoch += 1

		return batch_images, batch_classes, batch_boxes


if __name__ == '__main__':
	pascal = Pascal_voc('train')
	batch_images, batch_classes, batch_boxes = pascal.get_data()
	print(batch_images[0].shape)
	print(batch_classes[0].shape)
	print(batch_boxes[0].shape)