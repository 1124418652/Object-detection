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
		self.boundary1 = self.cell_size * self.cell_size * self.num_classes
		self.boundary2 = self.cell_size * self.cell_size * self.boxes_per_cell

		# the scale used in loss function
		self.object_scale = cfg.OBJECT_SCALE
		self.noobject_scale = cfg.NOOBJECT_SCALE
		self.class_scale = cfg.CLASS_SCALE
		self.coord_scale = cfg.COORD_SCALE

		# the super parameters used for model training 
		self.learning_rate = cfg.LEARNING_RATE
		self.batch_size = cfg.BATCH_SIZE
		self.alpha = cfg.ALPHA

		self.offset = np.transpose(np.reshape(np.array(
			[np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
			(self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

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
					


	def loss_calculate(self, logits, labels):
		pass

	def leaky_relu(self, alpha):
		def op(inputX):
			return tf.nn.leaky_relu(inputX, alpha=alpha, name='leaky_relu')
		return op



yolo = YOLO_V1_Net()