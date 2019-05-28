# -*- coding: utf-8 -*-
"""
Created on 2019/5/26

@Author: xhj
"""

import os
import numpy as np 
import tensorflow as tf 
import config as cfg
slim = tf.contrib.slim


class YOLO_V1_Net(object):

	def __init__(self, is_training=True):

		self.classes = cfg.CLASSES
		self.num_classes = len(self.classes)
		self.image_size = cfg.IMAGE_SIZE
		self.cell_size = cfg.CELL_SIZE
		self.boxes_per_cell = cfg.BOXES_PER_CELL
		self.output_size = (self.cell_size * self.cell_size)\
						   * (self.num_classes + 5 * self.boxes_per_cell)
		self.scale = 1.0 * self.image_size / self.cell_size   # the scale from image size to cell size

		# the scale used in loss function
		self.boundary1 = self.cell_size * self.cell_size * self.num_classes    # [0:boundary1] of output is classes of every boxes
		self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell # [boundary1:boundary2] of output is confidence of every boxes
		self.object_scale = cfg.OBJECT_SCALE
		self.noobject_scale = cfg.NOOBJECT_SCALE
		self.class_scale = cfg.CLASS_SCALE
		self.coord_scale = cfg.COORD_SCALE

		# the super parameters used for model training 
		self.learning_rate = cfg.LEARNING_RATE
		self.batch_size = cfg.BATCH_SIZE
		self.alpha = cfg.ALPHA
		# every grid has two bounding box, so the last dimension of offset should be 2
		self.offset_x = np.transpose(np.reshape(np.array(
			[np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
			(self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
		self.offset_y = np.transpose(self.offset_x, (1, 0, 2))

		self.inputX = tf.placeholder(
			tf.float32, [None, self.image_size, self.image_size, 3], name='input')
		self.logits = self.build_network(
			self.inputX, num_output=self.output_size, alpha=self.alpha,
			is_training=is_training)

		if is_training:
			self._labels = tf.placeholder(tf.float32, 
				[None, self.cell_size, self.cell_size, 5 + self.num_classes])
			self.loss = self.loss_calculate(self.logits, self._labels)
			self.total_loss = tf.losses.get_total_loss()
			tf.summary.scalar('total_loss', self.total_loss)

	def build_network(self, inputX, num_output=None, alpha=None, keep_prob=0.5,
					  is_training=True, scope='yolo-v1'):
		"""
		build the YOLO v1 network structure
		Args:
			inputX: 4-D tensor, the input of network, contains batches of images
			num_output: int type, the dimension of last fully connection layer
			alpha: float type, the parameter of leaky relu
			keep_prob: float type, the parameter used for dropout
			is_training: boolean type, training the parameters of W and b if true
			scope: str type, the name of this network
		Returns:
			logits: 4-D tensor, [batch_size, cell_size, cell_size, 30]
		"""

		with tf.variable_scope(scope):
			with slim.arg_scope(
				[slim.conv2d, slim.fully_connected],
				activation_fn=self.leaky_relu(alpha),
				weights_regularizer=slim.l2_regularizer(0.0005),
				weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)):

				with tf.variable_scope("group1"):
					net = tf.pad(inputX, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
							 name='pad_1')
					net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv1')
					net = slim.max_pool2d(net, 2, padding='SAME', scope='pool1')
				with tf.variable_scope("group2"):
					net = slim.conv2d(net, 192, 3, scope='conv2')
					net = slim.max_pool2d(net, 2, padding='SAME', scope='pool2')
				with tf.variable_scope("group3"):
					net = slim.conv2d(net, 128, 1, scope='conv3')
					net = slim.conv2d(net, 256, 3, scope='conv4')
					net = slim.conv2d(net, 256, 1, scope='conv5')
					net = slim.conv2d(net, 512, 3, scope='conv6')
					net = slim.max_pool2d(net, 2, padding='SAME', scope='pool3')
				with tf.variable_scope("group4"):
					net = slim.conv2d(net, 256, 1, scope='conv7')
					net = slim.conv2d(net, 512, 3, scope='conv8')
					net = slim.conv2d(net, 256, 1, scope='conv9')
					net = slim.conv2d(net, 512, 3, scope='conv10')
					net = slim.conv2d(net, 256, 1, scope='conv11')
					net = slim.conv2d(net, 512, 3, scope='conv12')
					net = slim.conv2d(net, 256, 1, scope='conv13')
					net = slim.conv2d(net, 512, 3, scope='conv14')
					net = slim.conv2d(net, 512, 1, scope='conv15')
					net = slim.conv2d(net, 1024, 3, scope='conv16')
					net = slim.max_pool2d(net, 2, padding='SAME', scope='pool4')
				with tf.variable_scope("group5"):
					net = slim.conv2d(net, 512, 1, scope='conv17')
					net = slim.conv2d(net, 1024, 3, scope='conv18')
					net = slim.conv2d(net, 512, 1, scope='conv19')
					net = slim.conv2d(net, 1024, 3, scope='conv20')
					net = slim.conv2d(net, 1024, 3, scope='conv21')
					net = slim.conv2d(net, 1024, 3, 2, scope='conv22')
				with tf.variable_scope("append_conv"):
					net = slim.conv2d(net, 1024, 3, scope='conv23')
					net = slim.conv2d(net, 1024, 3, scope='conv24')
				net = tf.transpose(net, [0, 3, 1, 2], name='channel_first')
				net = slim.flatten(net, scope='flatten_layer')
				with tf.variable_scope("fully_connect"):
					net = slim.fully_connected(net, 512, scope='fc1')
					net = slim.fully_connected(net, 4096, scope='fc2')
					net = slim.dropout(net, keep_prob=keep_prob, is_training=is_training,
									   scope='dropout_layer')
					# the last layer of network use linear activation function
					net = slim.fully_connected(net, num_output, activation_fn=None, scope='fc3')
		return net

	def iou_calculate(self, boxes1, boxes2, scope='iou_layer'):
		"""
		calculate the iou of two boxes
		Args:
			boxes1: 5-D tensor,[BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4], 
					the last dimension contains (x_center, y_center, w, h)
			boxes2: 5-D tensor
			scope: the name of this op
		"""

		with tf.variable_scope(scope):
			# transform (x_center, y_center, w, h) format to (x1, y1, x2, y2)
			boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.,
								 boxes1[..., 1] - boxes1[..., 3] / 2.,
								 boxes1[..., 0] + boxes1[..., 2] / 2.,
								 boxes1[..., 1] + boxes1[..., 3] / 2.],
								axis=-1)
			boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.,
								 boxes2[..., 1] - boxes2[..., 3] / 2.,
								 boxes2[..., 0] + boxes2[..., 2] / 2.,
								 boxes2[..., 1] + boxes2[..., 3] / 2.],
								axis=-1)

			# calculate the left up & right down points
			lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
			rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

			# calculate intersection
			wh_group = tf.maximum(0., rd - lu)
			intersection = wh_group[..., 0] * wh_group[..., 1]

			# calculate iou
			areas1 = boxes1[..., 2] * boxes1[..., 3]
			areas2 = boxes2[..., 2] * boxes2[..., 3]
			union_square = tf.maximum(areas1 + areas2 - intersection, 1e-10)
			iou = tf.clip_by_value(intersection / union_square, 0., 1., name='iou')
			return iou

	def loss_calculate(self, logits, labels, scope='loss_layer'):
		"""
		calculate the loss of YOLO v1
		Args:
			logits: 2-D tensor, the output of last fully-connect layer, [batch_size, 1470]
			labels: 4-D tensor, the label of input data
		"""

		with tf.variable_scope(scope):
			
			# extract prediction from network output
			predict_classes = tf.reshape(logits[:, :self.boundary1], 
				[self.batch_size, self.cell_size, self.cell_size, self.num_classes])
			# the confidence of 2 boxes
			predict_confidences = tf.reshape(logits[:, self.boundary1: self.boundary2],
				[self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
			predict_boxes = tf.reshape(logits[:, self.boundary2:],
				[self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
			
			# extract information from ground truth labels
			gt_confidence = tf.reshape(labels[..., 0], 
				[self.batch_size, self.cell_size, self.cell_size, 1])
			gt_classes = labels[..., 5:]
			gt_boxes = tf.reshape(labels[..., 1: 5], 
				[self.batch_size, self.cell_size, self.cell_size, 1, 4])   # 增加一个维度，为了与预测框计算iou时比较方便
			# 缩小至与预测框相同的比例
			gt_boxes = tf.tile(gt_boxes, multiples=[1, 1, 1, self.boxes_per_cell, 1]) / self.image_size

			# get the offset
			offset_x = tf.reshape(tf.constant(self.offset_x, dtype=tf.float32),
								  [1, self.cell_size, self.cell_size, self.boxes_per_cell])
			offset_x = tf.tile(offset_x, (self.batch_size, 1, 1, 1))
			offset_y = tf.reshape(tf.constant(self.offset_y, dtype=tf.float32),
								  [1, self.cell_size, self.cell_size, self.boxes_per_cell])
			offset_y = tf.tile(offset_y, (self.batch_size, 1, 1, 1))

			# modify the predict bounding boxes
			# predict_boxes中的中心坐标(x, y)是相对于当前grid cell左上角点的比例，w,h的值
			# 是实际w，h值归一化之后的平方根，所以分别需要进行加上offset和平方处理，使其与
			# label中的值只相差一个image_size的比例的线性变换
			predict_boxes_trans = tf.stack(
				[(predict_boxes[..., 0] + offset_x) / self.cell_size,     # predicted x center
				 (predict_boxes[..., 1] + offset_y) / self.cell_size,     # predicted y center
				 tf.square(predict_boxes[..., 2]),
				 tf.square(predict_boxes[..., 3])], axis=-1)

			# calculate iou
			predict_iou = self.iou_calculate(predict_boxes_trans, gt_boxes)

			# calculate I_obj tensor(the mask to separate obj and no-obj bboxes)
			# the shape of I_obj and iou are all [batch_size, cell_size, cell_size, boxes_per_cell]
			object_mask = tf.reduce_max(predict_iou, axis=3, keep_dims=True)
			object_mask = tf.cast(predict_iou >= object_mask, tf.float32) * gt_confidence

			# calculate I_noobj tensor
			noobject_mask = tf.ones_like(object_mask) - object_mask

			gt_boxes_tran = tf.stack(
				[gt_boxes[..., 0] * self.cell_size - offset_x,
				 gt_boxes[..., 1] * self.cell_size - offset_y，
				 tf.sqrt(gt_boxes[..., 2]),
				 tf.sqrt(gt_boxes[..., 3])], axis=-1)

			# calculate class loss, shape of classes: [batch_size, cell_size, cell_size, 20]
			class_delta = response * (predict_classes - gt_classes)
			class_loss = tf.reduce_mean(
				tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
				name='class_loss') * self.class_scale

			# calculate object loss, shape of confidence: [batch_size, cell_size, cell_size, boxes_per_cell]
			object_delta = object_mask * (predict_confidences - predict_iou)
			object_loss = tf.reduce_mean(
				tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), 
				name='object_loss') * self.object_scale


	def leaky_relu(self, alpha):
		def op(inputX):
			return tf.nn.leaky_relu(inputX, alpha=alpha, name='leaky_relu')
		return op



yolo = YOLO_V1_Net()