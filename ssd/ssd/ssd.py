#-*- coding: utf-8 -*-
import cv2
import config as cfg
import numpy as np 
import tensorflow as tf


class SSD(object):

	def __init__(self, threshold=0.6):
		"""
		构造函数
		Args:
			threshold: 判断是否保留先验框的阈值
		"""

		self.num_boxes = []
		self.feature_map_size = cfg.FEATURE_MAP_SIZE
		self.classes = cfg.CLASSES
		self.img_size = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
		self.num_classes = cfg.NUM_CLASSES  # 21

		# 6个特征图对应的参数
		self.feature_layers = cfg.FEATURE_LAYERS
		self.boxes_num = cfg.BOXES_NUM    # 每个像素点对应的先验框数目
		self.is_L2norm = cfg.IS_L2NORM
		self.n_boxes = cfg.N_BOXES        # 每个特征图的先验框数目

		# anchor 生成参数
		self.anchor_sizes = cfg.ANCHOR_SIZES
		self.anchor_ratios = cfg.ANCHOR_RATIOS

		# 网络输出的偏移量的缩放系数
		self.prior_scaling = cfg.PRIOR_SCALING
		self.threshold = threshold


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
				predictions.append(tf.nn.softmax(class_pred))
			return X, locations, predictions


	@staticmethod
	def ssd_anchor_layer(img_size, feature_map_size, anchor_size,
		anchor_ratio, anchor_step, box_num, offset=0.5):
		"""
		在单层 feature map 中生成锚框
		Args:
			img_size: 1-D list with 2 elements, (rows, cols)
			feature_map_size: 1-D list or tupple with 2 elements, (rows, cols)
			anchor_size: 锚框的尺寸信息
			anchor_ratio: 锚框缩放比例
			anchor_step: 该特征图与原图的比例
			box_num: 该特征图中每个像素点对应的锚框数目
			offset: 锚框中心相对于网格左上角点的偏移量
		"""
		y, x = np.mgrid[0:feature_map_size[0], 0:feature_map_size[1]]      # 生成网格左上角点
		y = (y.astype(np.float32) + offset) * anchor_step / img_size[0]    # 得到锚框在原图上的中点坐标
		x = (x.astype(np.float32) + offset) * anchor_step / img_size[1]

		# expand dims to support easy broadcasting in function ssd_decode()
		x = np.expand_dims(x, axis=-1)
		y = np.expand_dims(y, axis=-1)

		# calculate the width and height of every anchors
		h = np.zeros((box_num, ), dtype=np.float32)
		w = np.zeros((box_num, ), dtype=np.float32)

		h[0] = anchor_size[0] / img_size[0]
		w[0] = anchor_size[0] / img_size[1]
		h[1] = (anchor_size[0] * anchor_size[1]) ** 0.5 / img_size[0]
		w[1] = (anchor_size[0] * anchor_size[1]) ** 0.5 / img_size[1]

		for i, ratio in enumerate(anchor_ratio):
			h[i + 2] = anchor_size[0] / img_size[0] / (ratio ** 0.5)    # 从第三位开始添加
			w[i + 2] = anchor_size[0] / img_size[1] * (ratio ** 0.5)

		return x, y, w, h


	@staticmethod
	def ssd_decode(location, box, prior_scaling):
		"""
		对单层特征图的网络输出解码
		Args:
			location: 5-D tensor, 网络输出的预测坐标值, [batch_size, height, width, anchor_box, coordinations]
			box: 默认的锚框坐标及长宽，(x, y, w, h)
			prior_scaling: 先验框的缩放比例
		Returns:
			bboxes: 5-D tensor, 解码后的预测框， [batch_size, height, width, box, coordination]
					其中coordination表示的是预测框的左上角坐标和右下角坐标
		"""

		anchor_x, anchor_y, anchor_w, anchor_h = box 
		center_x = location[:, :, :, :, 0] * prior_scaling[0] * anchor_w + anchor_x 
		center_y = location[:, :, :, :, 1] * prior_scaling[1] * anchor_h + anchor_y
		w = anchor_w * tf.exp(location[:, :, :, :, 2] * prior_scaling[2])
		h = anchor_h * tf.exp(location[:, :, :, :, 3] * prior_scaling[3])
		bboxes = tf.stack([center_x - w / 2.0, center_y - h / 2.0, 
						   center_x + w / 2.0, center_y + h / 2.0],
						   axis=-1)
		return bboxes


	def choose_anchor_boxes(self, prediction, bboxes, threshold=None):
		"""
		对单层特征图的检验框进行阈值筛选，筛选出检验框所属类别的概率值>阈值的检验框
		Args:
			prediction: 5-D tensor, 网络输出的每个类别的预测概率,
						 [batch_size, height, width, boxes, classes]
			bboxes: 5-D tensor, 通过 ssd_decode() 解码得到的检验框
			threshold: 检验框所属类别概率的阈值
		"""

		if not threshold:
			threshold = self.threshold
		bboxes = tf.reshape(bboxes, [-1, 4])
		prediction = tf.reshape(prediction, [-1, self.num_classes])
		prediction = prediction[:, 1:]    # 提取除背景类外的所有类别
		max_classes = tf.argmax(prediction, axis=-1) + 1  # 提取索引,加1是因为从1开始
		scores = tf.reduce_max(prediction, axis=-1)       # 提取值

		filter_mask = scores > threshold
		max_classes = tf.boolean_mask(max_classes, filter_mask)
		scores = tf.boolean_mask(scores, filter_mask)
		bboxes = tf.boolean_mask(bboxes, filter_mask)

		return max_classes, scores, bboxes


	def nms(self, classes, scores, bboxes, top_k=-1, nms_threshold=0.5):
		"""
		对每一层特征图中筛选出的检验框进行非极大值抑制
		Args:
			classes: 1-D tensor, 检验框对应的类别, shape: [bboxes_num]
			scores: 1-D tensor, 检验框预测类别的概率, shape: [bboxes_num]
			bboxes: 2-D tensor, 检验框的坐标, shape: [bboxes_num, 4]
			nms_threshold: 判断两个检验框重合的iou阈值
		"""
		assert scores.shape[0] == bboxes.shape[0] == bboxes.shape[0] 
		keep_boxes = []
		indexes = np.argsort(-scores)     # 按置信度倒序排序，得到排序后的索引值
		classes = classes[indexes]
		scores = scores[indexes]
		bboxes = bboxes[indexes]



if __name__ == '__main__':
	ssd = SSD()
	X, locations, predictions = ssd.build_net()
	box = SSD.ssd_anchor_layer(ssd.img_size, (38, 38), (21, 45), [2, 0.5], 8, 4)
	boxes = ssd.ssd_decode(locations[0], box, ssd.prior_scaling)
	max_classes, scores, bboxes = ssd.choose_anchor_boxes(predictions[0], boxes)
	# ssd.nms(max_classes, scores, bboxes)
	print(boxes)