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


class Pascal_voc(object):

	def __init__(self, phase, rebuild=False):
		"""
		initialize the object
		Args:
			phase: str type, indicates the phase that dataset was used,
				   {'trian', 'validation'}
			rebuild: boolean type, if True, the cache file of labels will be rewrite
		"""

		self.devkit_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
		self.data_path = os.path.join(self.devkit_path, 'VOC2007')
		self.cache_path = cfg.CACHE_PATH
		self.batch_size = cfg.BATCH_SIZE
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.classes = cfg.CLASSES
		self.class2label = dict(zip(self.classes, range(len(self.classes))))
		self.flipped = cfg.FLIPPED         # append the flipped image to dataset or not
		self.phase = phase
		self.rebuild = rebuild
		self.cursor = 0                    # 记录当前读取的是gt_labels列表中的第几条数据
		self.epoch = 1                     # 记录完整读取gt_labels的次数
		self.gt_labels = None
		self.prepare()

	def load_image(self, image_path, flipped=False):
		"""
		load the image from image path which was input
		"""

		img = cv2.imread(image_path, 1)
		img = cv2.resize(img, (self.image_size, self.image_size))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = np.multiply(2/255., img) - 1.
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
		xml_path = self.data_path + '/Annotations/' + index + '.xml'
		img = cv2.imread(image_path)
		w_scale = self.image_size / img.shape[1]    # the scale of image's width
		h_scale = self.image_size / img.shape[0]    # the scale of image's height

		_label = np.zeros((self.cell_size, self.cell_size, 25))
		tree = ET.parse(xml_path)
		objs = tree.findall('object')
		for obj in objs:
			box = obj.find('bndbox')
			x1 = min(max(float(box.find('xmin').text)*w_scale, 0), self.image_size - 1)
			y1 = min(max(float(box.find('ymin').text)*h_scale, 0), self.image_size - 1)
			x2 = min(max(float(box.find('xmax').text)*w_scale, 0), self.image_size - 1)
			y2 = min(max(float(box.find('ymax').text)*h_scale, 0), self.image_size - 1)

			box = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
			cls_id = self.class2label[obj.find('name').text.lower().strip()]
			x_id = int(box[0] * 7 / self.image_size)  # the x index of bounding box
			y_id = int(box[0] * 7 / self.image_size)  # the y index of bounding box

			if _label[y_id, x_id, 0] == 1:
				continue

			_label[y_id, x_id, 0] = 1
			_label[y_id, x_id, 1:5] = box
			_label[y_id, x_id, 5 + cls_id] = 1

		return _label, len(objs)

	def load_labels(self):
		"""
		load the labels of all image
		"""

		cache_file = os.path.join(self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')
		if os.path.isfile(cache_file) and not self.rebuild:
			print("Loading gt_labels from: ", cache_file)
			with open(cache_file, 'rb') as fr:
				gt_labels = pickle.load(fr)
			return gt_labels

		print("Loading gt_labels from: ", self.data_path)    # else
		if not os.path.exists(self.cache_path):
			os.makedirs(self.cache_path)

		if 'train' == self.phase:
			txt_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')

		else:      # self.phase == 'validation'
			txt_name = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')

		with open(txt_name, 'r') as fr:
			self.image_indexes = [x.strip() for x in fr.readlines()]

		gt_labels = []
		for index in self.image_indexes:
			label, num = self.load_xml(index)
			if 0 == num:
				continue
			img_name = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
			gt_labels.append({'img_name': img_name,
							  'label': label,
							  'flipped': False})

		print('Saving gt_labels to: ' + cache_file)
		with open(cache_file, 'wb') as fw:     # 用缓存文件将label保存起来，下次就可以直接使用了
			pickle.dump(gt_labels, fw)

		return gt_labels

	def prepare(self):
		"""
		extract the groud truth labels from files to memory
		"""

		gt_labels = self.load_labels()
		if self.flipped:
			print("Appending horizontally-flipped training examples ...")
			gt_labels_cp = copy.deepcopy(gt_labels)
			for index in range(len(gt_labels_cp)):
				gt_labels_cp[index]['flipped'] = True
				# 图片反向之后，对应的网格图也应该反向
				gt_labels_cp[index]['label'] = gt_labels_cp[index]['label'][:, ::-1, :]
				# 每个网格中记录的目标的x坐标也应该反向
				for row in range(self.cell_size):
					for col in range(self.cell_size):
						if gt_labels_cp[index]['label'][row, col, 0] == 1:   # has object
							gt_labels_cp[index]['label'][row, col, 1] = \
								self.image_size - 1 - gt_labels_cp[index]['label'][row, col, 1]
			gt_labels.extend(gt_labels_cp)

		np.random.shuffle(gt_labels)
		self.gt_labels = gt_labels
		return gt_labels

	def get_data(self):
		batch_images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
		batch_labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 25))
		count = 0      # 记录batch中位置的变量
		while count < self.batch_size:
			img_name = self.gt_labels[self.cursor]['img_name']
			flipped = self.gt_labels[self.cursor]['flipped']
			batch_images[count, :, :, :] = self.load_image(img_name, flipped)
			batch_labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
			count += 1
			self.cursor += 1
			if self.cursor >= len(self.gt_labels):
				np.random.shuffle(self.gt_labels)
				self.cursor = 0
				self.epoch += 1

		return batch_images, batch_labels

if __name__ == '__main__':
	dataset = Pascal_voc('train')
	images, labels = dataset.get_data()
	print(images[0])