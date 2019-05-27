#-*- coding: utf-8 -*-
"""
Created on 27 March, 2019

Author: xhj
"""
import os
import cv2
import time
import logging
import argparse
import numpy as np 
from utils import *


def get_segmented_image(input_img, k, min_size, 
						neighborbood_8 = False):
	"""
	segment the input image

	Parameters:
	input_img: type of numpy.ndarray, the image that will be segmented
	k:  type of int, the parameter to contral the internal difference
	min_size: type of int, the minimize number of nodes in every regions
	neighborhood_8: type of boolean, use e neighborhood or not
	"""

	if not isinstance(input_img, np.ndarray):
		raise ValueError('The input image must be type of numpy.ndarray!')

	def diff(img, x1, y1, x2, y2):
		"""
		calculate the difference between two pixels

		Parameters:
		img: the input image
		x1, y1: the pixel location of node1
		x2, y2: the pixel location of node2
		"""

		bgr1 = img[y1, x1, :]
		bgr2 = img[y2, x2, :]
		I1 = bgr1.mean()
		I2 = bgr2.mean()

		out_diff = np.sum((bgr1 - bgr2) ** 2)# + (I1 - I2) ** 2 
		out_diff = np.sqrt(out_diff)
		return out_diff

	height, width = input_img.shape[: 2]  # as the parameters of function generate_image
	# num_nodes = input_img.size            # as the parameters of function segment_graph
	num_nodes = height * width
	print("Generating graph of input image...")
	begin = time.time()
	graph = build_graph(input_img, diff, neighborbood_8) # generate graph
	print("Finish generating graph. Time used: {0}s".format(time.time() - begin))
	
	print("Segmenting the image...")
	begin = time.time()
	forest = segment_graph(graph, num_nodes, k, min_size)
	print("Finish segmenting the imagae. Time used: {0}s".format(time.time() - begin))

	print("Generating image from segmented graph...")
	begin = time.time()
	segmented_img = generate_image(forest, width, height)
	print("Finish generating image. Time used: {0}s".format(time.time() - begin))

	return segmented_img


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description = 'Graph based segmentation')
	parser.add_argument('--sigma', type = float, default = 3.0, 
						help = 'the parameter for Gaussian Filter')
	parser.add_argument('--neighborhood_8', type = bool, default = True, 
						help = 'use 8 neighbor or not')
	parser.add_argument('--k', type = int, default = 10, 
						help = 'a constant to control the threshold function of the predicate')
	parser.add_argument('--min_size', type = int, default = 5000)

	args = parser.parse_args()

	img = cv2.imread('img/road1.jpg')	
	img = cv2.GaussianBlur(img, ksize = (7, 7), sigmaX = args.sigma)
	segmented_img = get_segmented_image(img, k = args.k, min_size = args.min_size)
	cv2.imwrite('img/road_segment.jpg', segmented_img)
	# cv2.waitKey(0)