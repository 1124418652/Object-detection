# -*- coding: utf-8 -*-
"""
Created on 2019/5/26

@Author: xhj
"""

import os
import cv2
import sys
import time
import datetime
import numpy as np 
import tensorflow as tf 
sys.path.append('../yolo/')
import config as cfg
sys.path.append('../utils/')
from pascal_voc import Pascal_voc
import timer
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
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.sess = None

        # the scale used in loss function
        self.boundary1 = self.cell_size * self.cell_size * self.num_classes    # [0:boundary1] of output is classes of every boxes
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell # [boundary1:boundary2] of output is confidence of every boxes
        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE
        # every grid has two bounding box, so the last dimension of offset should be 2
        self.offset_x = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))
        self.offset_y = np.transpose(self.offset_x, (1, 0, 2))

        # the super parameters used for model training 
        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        self.inputX = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3], name='input')
        self.logits, self.net1 = self.build_network(
            self.inputX, num_output=self.output_size, alpha=self.alpha,
            is_training=is_training)

        # if is_training:
        #     self._labels = tf.placeholder(tf.float32, 
        #         [None, self.cell_size, self.cell_size, 5 + self.num_classes])
        #     self.loss_calculate(self.logits, self._labels)
        #     self.total_loss = tf.losses.get_total_loss()
        #     tf.summary.scalar('total_loss', self.total_loss)


    def build_network(self, inputX=None, num_output=None, alpha=None, keep_prob=1,
                      is_training=True, scope='yolo'):
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

        if None == inputX:
            inputX = self.inputX
        if None == num_output:
            num_output = self.output_size
        if None == alpha:
            alpha = 0.1
        
        with tf.variable_scope('yolo'):
            with tf.variable_scope('conv_2'):
                net = self._conv_layer(inputX,  64, 7, 2,'conv_2')

            net1 = self._maxpool_layer(net,  2, 2)

            with tf.variable_scope('conv_4'):
                net = self._conv_layer(net1,  192, 3, 1,'conv_4')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_6'):
                net = self._conv_layer(net, 128, 1, 1,'conv_6')
            with tf.variable_scope('conv_7'):
                net = self._conv_layer(net, 256, 3, 1,'conv_7')
            with tf.variable_scope('conv_8'):
                net = self._conv_layer(net, 256, 1, 1,'conv_8')
            with tf.variable_scope('conv_9'):
                net = self._conv_layer(net, 512, 3, 1,'conv_9')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_11'):
                net3 = self._conv_layer(net, 256, 1, 1,'conv_11')
            with tf.variable_scope('conv_12'):
                net = self._conv_layer(net3, 512, 3, 1,'conv_12')
            with tf.variable_scope('conv_13'):
                net = self._conv_layer(net, 256, 1, 1,'conv_13')
            with tf.variable_scope('conv_14'):
                net = self._conv_layer(net, 512, 3, 1,'conv_14')
            with tf.variable_scope('conv_15'):
                net = self._conv_layer(net, 256, 1, 1,'conv_15')
            with tf.variable_scope('conv_16'):
                net = self._conv_layer(net, 512, 3, 1,'conv_16')
            with tf.variable_scope('conv_17'):
                net = self._conv_layer(net, 256, 1, 1,'conv_17')
            with tf.variable_scope('conv_18'):
                net = self._conv_layer(net, 512, 3, 1,'conv_18')
            with tf.variable_scope('conv_19'):
                net = self._conv_layer(net, 512, 1, 1,'conv_19')
            with tf.variable_scope('conv_20'):
                net = self._conv_layer(net, 1024, 3, 1,'conv_20')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_22'):
                net = self._conv_layer(net,  512, 1, 1,'conv_22')
            with tf.variable_scope('conv_23'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_30')
            with tf.variable_scope('conv_24'):
                net = self._conv_layer(net,  512, 1, 1,'conv_24')
            with tf.variable_scope('conv_25'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_25')
            with tf.variable_scope('conv_26'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_26')
            with tf.variable_scope('conv_28'):
                net = self._conv_layer(net,  1024, 3, 2,'conv_28')
            with tf.variable_scope('conv_29'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_29')
            with tf.variable_scope('conv_30'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_30')
            net = self._flatten(net)
            with tf.variable_scope('fc_33'):
                net = self._fc_layer(net,  512, activation=self.leak_relu,scope='fc_33')
            with tf.variable_scope('fc_34'):
                net = self._fc_layer(net, 4096, activation=self.leak_relu,scope='fc_34')
            with tf.variable_scope('fc_36'):
                net = self._fc_layer(net, 7*7*30,scope='fc_36')
        return net, net3

    def _conv_layer(self, x, num_filters, filter_size, stride,scope):

        in_channels = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size,
                                                  in_channels, num_filters], stddev=0.1),name='weights')
        bias = tf.Variable(tf.zeros([num_filters,]),name='biases')

        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID",name=scope)
        output = self.leak_relu(tf.nn.bias_add(conv, bias))

        return output

    def leak_relu(self,x, alpha=0.1):
        return tf.maximum(alpha * x, x)

    def _fc_layer(self, x,  num_out, activation=None,scope=None):

        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1),name='weights')
        bias = tf.Variable(tf.zeros([num_out,]),name='biases')
        output = tf.nn.xw_plus_b(x, weight, bias,name=scope)
        if activation:
            output = activation(output)

        return output

    def _maxpool_layer(self, x,  pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                                strides=[1, stride, stride, 1], padding="SAME")

        return output

    def _flatten(self, x):
        """flatten the x"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channle first mode
        nums = np.product(x.get_shape().as_list()[1:])
        return tf.reshape(tran_x, [-1, nums])


###############################################################
# detector
###############################################################


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

            # calculate iou（这里的iou就是confidence，即bounding box中包含目标的概率）
            predict_iou = self.iou_calculate(predict_boxes_trans, gt_boxes)

            # calculate I_obj tensor(the mask to separate obj and no-obj bboxes)
            # the shape of I_obj and iou are all [batch_size, cell_size, cell_size, boxes_per_cell]
            object_mask = tf.reduce_max(predict_iou, axis=3, keepdims=True)
            object_mask = tf.cast(predict_iou >= object_mask, tf.float32) * gt_confidence

            # calculate I_noobj tensor
            noobject_mask = tf.ones_like(object_mask) - object_mask

            gt_boxes_tran = tf.stack(
                [gt_boxes[..., 0] * self.cell_size - offset_x,
                 gt_boxes[..., 1] * self.cell_size - offset_y,
                 tf.sqrt(gt_boxes[..., 2]),
                 tf.sqrt(gt_boxes[..., 3])], axis=-1)

            # calculate class loss, shape of classes: [batch_size, cell_size, cell_size, 20]
            class_delta = gt_confidence * (predict_classes - gt_classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # calculate object loss, shape of confidence: [batch_size, cell_size, cell_size, boxes_per_cell]
            object_delta = object_mask * (predict_confidences - predict_iou)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), 
                name='object_loss') * self.object_scale

            # calculate noobject loss
            noobject_delta = noobject_mask * predict_confidences
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # calculate coordinate loss
            # shape of coordinate: [batch_size, cell_size, cell_size, boxes_per_cell, 4]
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - gt_boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            # tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            # tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            # tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            # tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])

            return logits, class_delta


    def leaky_relu(self, alpha):
        def op(inputX):
            return tf.nn.leaky_relu(inputX, alpha=alpha, name='leaky_relu')
        return op


    def boxes_filter(self, logits):
        """
        select the right boxes of object from network's output
        Args:
            logits: the output of the network
        """

        predict_classes = tf.reshape(logits[0, 0: self.boundary1], [7, 7, 20])
        predict_confidence = tf.reshape(logits[0, self.boundary1: self.boundary2], [7, 7, 2])
        predict_boxes = tf.reshape(logits[0, self.boundary2:], [7, 7, 2, 4])
        offset_x = tf.constant(self.offset_x, dtype=tf.float32)
        offset_y = tf.constant(self.offset_y, dtype=tf.float32)
        predict_boxes_trans = tf.stack(
            [(predict_boxes[..., 0] + offset_x) / self.cell_size * self.image_size,
             (predict_boxes[..., 1] + offset_y) / self.cell_size * self.image_size,
             tf.square(predict_boxes[..., 2]) * self.image_size,
             tf.square(predict_boxes[..., 3]) * self.image_size], axis=3)

        # calculate the probability of every classes(because the predict probability of classes
        # that output by network is conditional probability P(class|confidence))
        scores = tf.expand_dims(predict_confidence, axis=-1)\
                 * tf.expand_dims(predict_classes, axis=2)
        scores = tf.reshape(scores, [-1, 20])
        boxes = tf.reshape(predict_boxes_trans, [-1, 4])

        # 拿到每个box的类别和得分
        box_classes = tf.argmax(scores, axis=1)
        box_classes_scores = tf.reduce_max(scores, axis=1)

        # 过滤掉得分小于阈值的box
        filter_mask = box_classes_scores > self.threshold
        scores = tf.boolean_mask(box_classes_scores, filter_mask)
        boxes = tf.boolean_mask(boxes, filter_mask)
        box_classes = tf.boolean_mask(box_classes, filter_mask)

        # 调整检测得到的方框的表示方式，(x_center, y_center, w, h) -> (x1, y1, x2, y2)
        _boxes = tf.stack(
            [boxes[:, 1] - 0.5 * boxes[:, 3],   # x1 
             boxes[:, 0] - 0.5 * boxes[:, 2],   # y1
             boxes[:, 1] + 0.5 * boxes[:, 3],   # x2
             boxes[:, 0] + 0.5 * boxes[:, 2]],  # y2
             axis=1)

        # nms
        nms_indices = tf.image.non_max_suppression(_boxes, scores, 10, self.iou_threshold)

        scores = tf.gather(scores, nms_indices)
        boxes = tf.gather(boxes, nms_indices)
        box_classes = tf.gather(box_classes, nms_indices)

        return scores, boxes, box_classes

    def _draw_rectangle(self, image, result):
        """
        label the object which was detected in source image
        Args:
            image: 3-D array, the source image(this image will be modified)
            result: 2-D array, [[x_center, y_center, w, h, class, score],...]
        """
        
        for i in range(len(result)):
            x_center, y_center, w, h = result[i][0: 4]
            cv2.rectangle(image, (int(x_center - w / 2), int(y_center - h / 2)),
                (int(x_center + w / 2), int(y_center + h / 2)), (0, 255, 0), 2)
            cv2.rectangle(image, (int(x_center - w / 2), int(y_center - h / 2 - 20)),
                (int(x_center + w / 2), int(y_center - h / 2)), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            cv2.putText(
                image, result[i][4] + ': %.2f'%(result[i][5]),
                (int(x_center - w / 2 + 2), int(y_center - h / 2 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (0, 0, 0), 1, lineType)


    def init_detector(self, weight_file = None):
        """
        initialize the detector of Yolo network
        """

        if not weight_file:
            weight_file = '../weights/YOLO_small.ckpt'

        if self.sess:
            self.sess.close()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        print('Restoring weights from: ' + weight_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, weight_file)
        print('Finishing restoring weight file.')
        

    def image_detect(self, image, one_img=True):
        """
        detect object in input images
        Args:
            image: 3-D array, shape of [image_size, image_size, 3]
            one_img: boolean type, 是否只需要检测一次
        """
        if 3 != image.ndim:
            raise ValueError("Must input 3-D array image")

        tmp_image = image.copy()
        height, width = image.shape[:2]
        tmp_image = cv2.resize(tmp_image, (self.image_size, self.image_size)).astype(np.float32)
        tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_BGR2RGB) / 255. * 2. - 1.
        tmp_image = np.reshape(tmp_image, (1, self.image_size, self.image_size, 3))

        h_scale = height / self.image_size
        w_scale = width / self.image_size
        if not self.sess:
            raise ValueError("Initialize the detector first!")

        _scores, _boxes, _box_classes = self.boxes_filter(self.logits)
        scores, boxes, box_classes = self.sess.run(
            [_scores, _boxes, _box_classes], 
            feed_dict = {self.inputX: tmp_image})
        print(scores)

        boxes = np.stack([
            boxes[..., 0] * w_scale,
            boxes[..., 1] * h_scale,
            boxes[..., 2] * w_scale,
            boxes[..., 3] * h_scale],
            axis=1)

        box_classes = [self.classes[x] for x in box_classes]
        result = []
        for i in range(len(scores)):
            tmp = list(map(int, boxes[i].tolist()))
            tmp.append(box_classes[i])
            tmp.append(scores[i])
            result.append(tmp)
        # result = np.array(result)
        # print(result)
        self._draw_rectangle(image, result)

        if one_img:
            self.sess.close()

        return image


    def video_detector(self, from_camera = True, file_path = None):
        """
        detect object from camera
        Args:
            from_camera: boolean type, the video is from camera if true
            file_path: str type, the path of video file(parameter from_camera is False)
        """

        if from_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("can't open the camera")
            ret, frame = cap.read()
            cv2.namedWindow('image windows', 2)
            while ret:
                image = self.image_detect(frame, False)
                cv2.imshow('image windows', image)
                if 27 == cv2.waitKey(1):
                    break
                ret, frame = cap.read()
            cv2.destroyAllWindows()

        else:
            try:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
            except:
                raise ValueError("can't open the file: {0}".format(file_path))

            cv2.namedWindow("image windows", 2)
            while ret:
                image = self.image_detect(frame, False)
                cv2.imshow('image windows', image)
                if 27 == cv2.waitKey(1):
                    break
                ret, frame = cap.read()
            cv2.destroyAllWindows()

        self.sess.close()



######################################################################
# solver
######################################################################
    def init_solver(self, train_dataset, weight_file=None, append_name=None):
        """
        initialize the Solver of Yolo network
        """

        self.data = train_dataset
        if not weight_file:
            weight_file = '../weights/YOLO_small.ckpt'
        if not append_name:
            append_name = 'None'

        # loss计算的参数的初始化
        self._labels = tf.placeholder(tf.float32, 
                [None, self.cell_size, self.cell_size, 5 + self.num_classes])


        self.loss_logits, self.loss_class_delta = self.loss_calculate(self.logits, self._labels)
        

        self.total_loss = tf.losses.get_total_loss()
        tf.summary.scalar('total_loss', self.total_loss)
        
        self.summary_iter = cfg.SUMMARY_ITER
        self.save_iter = cfg.SAVE_ITER
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, append_name, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 模型保存的参数
        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(max_to_keep = 5)
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        # 学习率衰减的参数
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.max_iter = cfg.MAX_ITER       # 最多训练的次数
        self.initial_learning_rate = cfg.LEARNING_RATE      # 起始的学习率
        self.decay_steps = cfg.DECAY_STEP
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        """
        tf.train.exponential_decay(
            learning_rate,               初始学习率
            global_step,                 当前迭代次数
            decay_steps,        衰减速度（在迭代到该次数时学习率衰减为learning_rate * decay_rate^(global_step/decay_steps)）
            decay_rate,         学习率衰减系数，通常介于0-1之间。
            staircase=False,    (默认值为False,当为True时，（global_step/decay_steps）则被转化为整数) ,选择不同的衰减方式。
            name=None)
        """
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps, 
            self.decay_rate, self.staircase, name='learning_rate')
        
        # 优化器
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss, self.global_step)

        # 配置session
        gpu_options = tf.GPUOptions()
        config = tf.ConfigProto(gpu_options=gpu_options)
        if self.sess:
            self.sess.close()
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        # 从文件中提取预训练的参数值
        print('Restoring weights from: ' + weight_file)
        self.saver.restore(self.sess, weight_file)
        print('Finish restoring weights.')

        self.writer.add_graph(self.sess.graph)


    def train(self):
        # train_timer = timer.Timer()
        # data_load_timer = timer.Timer()

        for step in range(1, self.max_iter + 1):

            # data_load_timer.tic()
            begin = time.time()
            images, labels = self.data.get_data()
            # print(images.shape)
            # w1 = self.sess.graph.get_operation_by_name('yolo/conv_2/weights')
            # print(w1.run(session=self.sess))
            # load_timer.toc()
            feed_dict = {self.inputX: images,
                         self._labels: labels}
            loss, _ = self.sess.run([self.total_loss, self.train_op], feed_dict=feed_dict)
            # print(self.loss_logits.eval(session=self.sess, feed_dict=feed_dict))
            # print(self.net1.eval(session=self.sess, feed_dict=feed_dict))
            min_loss = 100
            if step % self.summary_iter == 0:
                if step % (self.summary_iter * 10) == 0:
                    # train_timer.tic()
                    summary_str = self.sess.run(self.summary_op,
                        feed_dict=feed_dict)
                    # train_timer.toc()

                    log_str = """{} Epoch: {}, Step: {}, Learning rate: {}, \nLoss: {:5.3f}\nSpeed: {:.3f} s/100step""".format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        self.data.epoch, step, round(self.learning_rate.eval(session=self.sess),6),
                        loss, (time.time()-begin)/1000)

                    print(log_str)

                else:
                    summary_str, _ = self.sess.run(
                            [self.summary_op, self.train_op],
                            feed_dict=feed_dict)
                self.writer.add_summary(summary_str, step)

            if step > self.save_iter and loss < min_loss:
                min_loss = loss
                print("{} Saving checkpoint file to: {}".format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step)
            



if __name__ == '__main__':
    yolo = YOLO_V1_Net()
    # print(yolo.logits)
    yolo.init_detector()
    dataset = Pascal_voc('train')
    # yolo.init_solver(dataset)
    image = cv2.imread("../test-images/4.jpg")
    # dataset = Pascal_voc('train')
    # image, labels = dataset.get_data()
    # image = ((image[0] + 1) * 255/2).astype(np.uint8)
    # print(image)
    # cv2.imshow("image", image)
    image = yolo.image_detect(image)
    cv2.imshow("si", image)
    cv2.waitKey(0)
    # yolo.video_detector(True, '../test-images/轨道有异物2.mp4')
    # dataset = Pascal_voc('train')
    # yolo.init_solver(dataset, append_name='pascal_voc_trained_weight')
    # yolo.train()