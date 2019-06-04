#-*- coding: utf-8 -*-
import cv2
import config as cfg
import numpy as np 
import tensorflow as tf


class SSD(object):

	def __init__(self):
		
		self.num_boxes = []
		self.feature_map_size = cfg.FEATURE_MAP_SIZE
		self.classes = cfg.CLASSES
		self.img_size = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
		self.num_classes = cfg.NUM_CLASSES

		self.feature_layers = cfg.FEATURE_LAYERS
		self.boxes_num = cfg.BOXES_NUM
		self.is_L2norm = cfg.IS_L2NORM

		# anchor 生成参数
		self.anchor_sizes = cfg.ANCHOR_SIZES
		self.anchor_ratios = cfg.ANCHOR_RATIOS


	def l2norm(self, X, trainable=True, scope='L2Normalization'):
		n_channles = X.get_shape().as_list()[-1]   # 获取特征图的通道数
		l2_norm = tf.nn.l2_normalize(X, axis=[3], epsilon=1e-12)    # 只在channels维度上做归一化
		with tf.variable_scope(scope):
			gamma = tf.get_variable('gamma', shape=[n_channles,], dtype=tf.float32,
				trainable=trainable)
			return l2_norm * gamma


	def feat_map_prediction(self, X, num_classes, boxes_num, is_L2norm, scope='multibox'):
		"""
		从一层 feature map 中预测坐标和类别
		Args:
			X: 4-D tensor, [batch_num, width, height, channels]
			num_classes: int type, 21
			boxes_num: int type, 该层特征图中
			is_L2norm: boolean, 表示是否需要进行L2归一化
		"""
		reshape = [-1] + X.get_shape().as_list()[1:-1]  # 只需要feature map的width和height
		with tf.variable_scope(scope):
			if is_L2norm:
				X = self.l2norm(X)
			# 先预测位置 --> x, y, w, h    回归问题
			location_pred = self.conv2d(X, filter=boxes_num * 4, ksize=[3, 3],
				activation=None, scope='conv_loc')   # 不需要激活函数
			location_pred = tf.reshape(location_pred, reshape + [boxes_num, 4])
			# 预测类别   --> class   分类问题，需要经过softmax层
			class_pred = self.conv2d(X, filter=boxes_num * num_classes, ksize=[3, 3],
				activation=None, scope='conv_cls')
			class_pred = tf.reshape(class_pred, reshape + [boxes_num, num_classes])
			return location_pred, class_pred


	def conv2d(self, X, filter, ksize, stride=[1,1], padding='same',
		dilation=[1,1], activation=tf.nn.relu, scope='conv2d'):
		"""
		执行卷积运算
		Args:
			X: 4-D tensor
			filter: int 类型，卷积核的数目
			ksize: [int, int]，卷积核的尺寸
			dilation: 卷积核的扩张率
		"""
		return tf.layers.conv2d(inputs=X, filters=filter, kernel_size=ksize, 
			strides=stride, dilation_rate=dilation, padding=padding, name=scope)


	def max_pool2d(self, X, psize, stride, scope='max_pool2d'):
		return tf.layers.max_pooling2d(inputs=X, pool_size=psize,
			strides=stride, name=scope, padding='same')


	def pad2d(self, X, pad):
		return tf.pad(X, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])


	def dropout(self, X, d_rate=0.5):
		return tf.layers.dropout(inputs=X, rate=d_rate)


	def build_net(self):
		"""
		构建SSD的基础网络架构
		"""

		check_points = {}  # 用于提取第4,7~11层的feature map
		X = tf.placeholder(dtype=tf.float32, shape=[None, 300, 300, 3])
		with tf.variable_scope('ssd_300_vgg'):
			# group1
			net = self.conv2d(X, 64, [3, 3], scope='conv1_1')
			net = self.conv2d(net, 64, [3, 3], scope='conv1_2')
			net = self.max_pool2d(net, psize=[2, 2], stride=[2, 2], scope='pool1')

			# group2
			net = self.conv2d(net, 128, [3, 3], scope='conv2_1')
			net = self.conv2d(net, 128, [3, 3], scope='conv2_2')
			net = self.max_pool2d(net, psize=[2, 2], stride=[2, 2], scope='pool2')

			# group3
			net = self.conv2d(net, 256, [3, 3], scope='conv3_1')
			net = self.conv2d(net, 256, [3, 3], scope='conv3_2')
			net = self.conv2d(net, 256, [3, 3], scope='conv3_3')
			net = self.max_pool2d(net, psize=[2, 2], stride=[2, 2], scope='pool3')

			# group4
			net = self.conv2d(net, 512, [3, 3], scope='conv4_1')
			net = self.conv2d(net, 512, [3, 3], scope='conv4_2')
			net = self.conv2d(net, 512, [3, 3], scope='conv4_3')
			check_points['block4'] = net
			net = self.max_pool2d(net, psize=[2, 2], stride=[2, 2], scope='pool4')

			# group5
			net = self.conv2d(net, 512, [3, 3], scope='conv5_1')
			net = self.conv2d(net, 512, [3, 3], scope='conv5_2')
			net = self.conv2d(net, 512, [3, 3], scope='conv5_3')
			net = self.max_pool2d(net, psize=[3, 3], stride=[1, 1], scope='pool5')

			# group6
			# 采用了空洞卷积
			net = self.conv2d(net, 1024, [3, 3], dilation=[6, 6], scope='conv6')

			# group7
			net = self.conv2d(net, 1024, [3, 3], scope='conv7')
			check_points['block7'] = net

			# group8
			net = self.conv2d(net, 256, [1, 1], scope='conv8_1x1')
			net = self.conv2d(self.pad2d(net, 1), 512, [3, 3], stride=[2, 2], padding='valid',
				scope='conv8_3x3')
			check_points['block8'] = net

			# group9
			net = self.conv2d(net, 128, [1, 1], scope='conv9_1x1')
			net = self.conv2d(self.pad2d(net, 1), 256, [3, 3], stride=[2, 2], padding='valid',
				scope='conv9_3x3')
			check_points['block9'] = net

			# group10
			net = self.conv2d(net, 128, [1, 1], scope='conv10_1x1')
			net = self.conv2d(net, 256, [3, 3], padding='valid', scope='conv10_3x3')
			check_points['block10'] = net

			# group11
			net = self.conv2d(net, 128, [1, 1], scope='conv11_1x1')
			net = self.conv2d(net, 256, [3, 3], padding='valid', scope='conv11_3x3')
			check_points['block11'] = net

			locations = []
			predictions = []
			for index, layer in enumerate(self.feature_layers):
				location_pred, class_pred = self.feat_map_prediction(
					X = check_points[layer],
					num_classes=self.num_classes,
					boxes_num=self.boxes_num[index],
					is_L2norm=self.is_L2norm[index], 
					scope=layer + '_box')
				locations.append(location_pred)
				predictions.append(class_pred)
			return X, locations, predictions


	@staticmethod
	def ssd_anchor_layer(img_size, feature_map_size, anchor_size,
		anchor_ratio, anchor_step, box_num, offset=0.5):
		"""
		在一层 feature map 中生成锚框
		Args:
			img_size: 1-D list with 2 elements, (rows, cols)
			feature_map_size: 1-D list or tupple with 2 elements, (rows, cols)
		"""

if __name__ == '__main__':
	ssd = SSD()
	X, locations, predictions = ssd.build_net()
	print(locations)