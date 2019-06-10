#-*- coding: utf-8 -*-
"""
Created on 2019/5/29

@Author: xhj
"""

import os 
import cv2
import argparse
from yolo.yolo_v1 import YOLO_V1_Net


def image_detection(img_path='test-images/1.jpg'):
	if not os.path.exists(img_path):
		raise ValueError("Image {0} don't is not found!".format(img_path))
	yolo = YOLO_V1_Net()
	yolo.init_detector('weights/YOLO_small.ckpt')
	image = cv2.imread(img_path)
	image_detected = yolo.image_detect(image)
	cv2.imshow("image", image_detected)
	cv2.waitKey(0)

def video_detection():
	pass

if __name__ == '__main__':
	image_detection()