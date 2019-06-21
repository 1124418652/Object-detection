#-*- coding: utf-8 -*-
"""
Created on 2019/6/10

@Author: xhj
"""

import os
import cv2
import sys
import time
import datetime
import config as cfg
import numpy as np 
import tensorflow as tf

sys.path.append('../utils/')
from pascal_voc import Pascal_voc


class SSD(object):

    def __init__(self, threshold=0.2):
        """
        构造函数
        Args:
            threshold: 判断是否保留先验框的阈值
        """

        self.feature_map_size = cfg.FEATURE_MAP_SIZE
        self.classes_name = cfg.CLASSES
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
        self.anchor_steps = cfg.ANCHOR_STEPS

        # 网络输出的偏移量的缩放系数
        self.prior_scaling = cfg.PRIOR_SCALING
        self.threshold = threshold
        self.show_info = []

        self.sess = None


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
            strides=stride, dilation_rate=dilation, padding=padding, name=scope,
            activation=activation)


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
            net = self.conv2d(net, 1024, [1, 1], scope='conv7')
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
            logits = []
            if 0 < len(self.show_info):
                self.show_info = []
            for index, layer in enumerate(self.feature_layers):
                self.show_info.append("Size of layer %(index)s's feature map: %(size)s"%{'index':layer, 
                    'size':check_points[layer].get_shape().as_list()})
                location_pred, class_pred = self.feat_map_prediction(
                    X = check_points[layer],
                    num_classes=self.num_classes,
                    boxes_num=self.boxes_num[index],
                    is_L2norm=self.is_L2norm[index], 
                    scope=layer + '_box')
                locations.append(location_pred)
                predictions.append(tf.nn.softmax(class_pred))
                logits.append(class_pred)
            return X, locations, predictions, logits


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
        Returns:
            x: 2-D ndarray, shape [feat_map_size, feat_map_size], 表示特征图中每个像素
               位置对应的中心坐标x
            y: 2-D ndarray, 表示特征图中每个像素位置对应的中心坐标y
            w, h: 1-D ndarray, shape [box_num], 同一特征图中所有像素位置的锚框的数目以及w和h都是
                  一样的，所以只需要用一维数组来表示即可
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
    def ssd_decode(location, anchor, prior_scaling):
        """
        对单层特征图的网络输出解码
        Args:
            location: 5-D tensor, 网络输出的预测坐标值, [batch_size, height, width, anchor_box, coordinations]
            anchor: 默认的锚框坐标及长宽，(x, y, w, h)
            prior_scaling: 先验框的缩放比例
        Returns:
            bboxes: 5-D tensor, 解码后的预测框， [batch_size, height, width, box, coordination]
                    其中coordination表示的是预测框的左上角坐标和右下角坐标
        """

        anchor_x, anchor_y, anchor_w, anchor_h = anchor 
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
        对单层特征图的检验框进行阈值筛选，筛选出检验框所属类别的概率值>阈值的检验框，调用该函数时有个前提条件是batch
        size 维度要么没有，要么值为1
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


    def get_result(self, locations, predictions):
        """
        从SSD网络的输出中筛选得到完整的classes，scores，bboxes结果，有个前提条件是batch_size必须为1
        Args:
            locations: build_network()的输出，从6个特征图中预测得到的bboxes坐标
            predictions: build_network()的输出，从6个特征图中预测得到的每个bboxes对应的类别概率
        """

        layers_anchors = []
        classes_list = []
        scores_list = []
        bboxes_list = []

        # 分别计算每一层特征图的anchor boxes
        for index, size in enumerate(self.feature_map_size):
            anchors = self.ssd_anchor_layer(self.img_size, size, 
                                           self.anchor_sizes[index],
                                           self.anchor_ratios[index],
                                           self.anchor_steps[index],
                                           self.boxes_num[index],
                                           offset=0.5)
            layers_anchors.append(anchors)
        
        # 分别计算predict boxes, classes 以及 scores
        for i in range(6):
            predict_boxes = self.ssd_decode(locations[i], layers_anchors[i], 
                self.prior_scaling)
            max_classes, scores, bboxes = self.choose_anchor_boxes(predictions[i], 
                predict_boxes, self.threshold)
            classes_list.append(max_classes)
            scores_list.append(scores)
            bboxes_list.append(bboxes)

        # 将不同特征图的预测结果进行拼接
        classes = tf.concat(classes_list, axis=0)
        scores = tf.concat(scores_list, axis=0)
        bboxes = tf.concat(bboxes_list, axis=0)
        return classes, scores, bboxes


    @staticmethod
    def nms(classes, scores, bboxes, top_k=-1, nms_threshold=0.5):
        """
        对每一层特征图中筛选出的检验框进行非极大值抑制
        Args:
            classes: 1-D tensor, 检验框对应的类别, shape: [bboxes_num]
            scores: 1-D tensor, 检验框预测类别的概率, shape: [bboxes_num]
            bboxes: 2-D tensor, 检验框的坐标, shape: [bboxes_num, 4]
            nms_threshold: 判断两个检验框重合的iou阈值
        """
        assert scores.shape[0] == bboxes.shape[0] == bboxes.shape[0] 
        indexes = np.argsort(-scores)     # 按置信度倒序排序，得到排序后的索引值
        classes = classes[indexes][:top_k]
        scores = scores[indexes][:top_k]
        bboxes = bboxes[indexes][:top_k]
        keep_boxes = np.ones(scores.shape, dtype=np.bool)
        for i in range(scores.shape[0] - 1):
            if keep_boxes[i]:
                ious = SSD.calculate_iou(bboxes[i], bboxes[i+1:])
                keep_overlap = np.logical_or(ious < nms_threshold, classes[i] != classes[i+1:])
                keep_boxes[i+1:] = np.logical_and(keep_boxes[i+1:], keep_overlap)
        indexes_selected = np.where(keep_boxes)
        return classes[indexes_selected], scores[indexes_selected], bboxes[indexes_selected]


    @staticmethod
    def calculate_iou(boxes1, boxes2):
        """
        计算两组候选框之间的iou
        Args:
            boxes1: 2-D ndarray, [1, 4](xmin, ymin, xmax, ymax)
            boxes2: 2-D ndarray, [num_boxes, 4](xmin, ymin, xmax, ymax)
        Returns:
            iou: 1-D ndarray, [num_boxes] 
        """
        if not boxes1.ndim == 2:
            boxes1 = np.expand_dims(boxes1, axis = 0)
        if not boxes2.ndim == 2:
            boxes2 = np.expand_dims(boxes2, axis = 0)

        xmin = np.maximum(boxes1[:, 0], boxes2[:, 0])
        ymin = np.maximum(boxes1[:, 1], boxes2[:, 1])
        xmax = np.minimum(boxes1[:, 2], boxes2[:, 2])
        ymax = np.maximum(boxes1[:, 3], boxes2[:, 3])

        # 计算两组矩形框的交集
        height = np.maximum(ymax - ymin, 0.)
        width = np.maximum(xmax - xmin, 0.)
        inter_area = height * width
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        iou = inter_area / (area1 + area2 - inter_area)
        return iou


class SSD_DETECTOR(SSD):

    def __init__(self, weight_file='../weight/ssd_vgg_300_weights.ckpt'):
        """
        初始化SSD检测网络所需的变量
        Args:
            weight_file: SSD 网络权重文件的路径
        """
        SSD.__init__(self, 0.2)
        self.inputX, self.locations, self.predictions, _ = self.build_net()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)      # 创建会话
        self.sess.run(tf.global_variables_initializer())

        # if not os.path.exists(weight_file):
        #   raise ValueError("file {filename} is not existed.".format(filename = weight_file))
        
        # 导入模型的权重文件
        print("\nStart loading weight file...")
        begin = time.time()
        saver = tf.train.Saver()
        saver.restore(self.sess, weight_file)
        print("Finishing loading weight file, time used: %.3f s\n"%(time.time()-begin))

        # 网络变量初始化
        print("Start initializing parameters of network...")
        begin = time.time()
        print("size of network input: ", self.inputX.get_shape().as_list())
        for info in self.show_info:
            print(info)
        self.classes, self.scores, self.pboxes = self.get_result(self.locations, 
            self.predictions)
        print("Finish initializing parameters. time used: %.3f s\n"%(time.time()-begin))


    def _draw_rectangle(self, image, classes, scores, bboxes, 
        colors=(0, 0, 255), thickness=2):
        """
        在输入图片中绘制检测出的目标框
        Args:
            image: 3-D array, the image to detect
            classes: 1-D array, the classes of every detected boxes
            scores: 1-D array, the scores of every detected boxes
            bboxes: 2-D array, shape of [num_boxes, 4], the coordinations of boxes
        """
        if not 3 == image.ndim:
            raise ValueError

        height, width = image.shape[:2]
        objs = []
        for i in range(classes.shape[0]):
            box = bboxes[i]
            class_name = self.classes_name[classes[i]-1]
            score = scores[i]
            info = "%s: %.3f" % (class_name, score)
            objs.append(info)
            minx = int(box[0] * width)
            miny = int(box[1] * height)
            maxx = int(box[2] * width)
            maxy = int(box[3] * height)
            cv2.rectangle(image, (minx, miny), (maxx, maxy), colors, thickness)
            cv2.rectangle(image, (minx, miny - 20), (maxx, miny), (125, 125, 125), -1)
            cv2.putText(image, info, (minx - 2, miny - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                (0, 255, 0), 1)
        return objs


    def image_detect(self, image, one_img=True):
        """
        detect objects in input image
        Args:
            image: 3-D ndarray, input image, rgb format
            one_img: boolean type, 如果为True，在该函数中关闭会话
        """

        if 3 != image.ndim:
            raise ValueError("Must input 3-D array image!")
        
        class Obj_result():
            """
            用于保存单幅图片中目标检测结果的类
            """
            
            @property
            def classes(self):
                return self._classes

            @classes.setter
            def classes(self, classes):
                if not  1 == classes.ndim:
                    raise ValueError
                self._classes = classes

            @property
            def scores(self):
                return self._scores

            @scores.setter
            def scores(self, scores):
                if not 1 == scores.ndim:
                    raise ValueError
                self._scores = scores
            
            @property
            def bboxes(self):
                return self._bboxes
            
            @bboxes.setter
            def bboxes(self, bboxes):
                if not 4 == bboxes.shape[-1]:
                    raise ValueError
                if 1 == bboxes.ndim:
                    bboxes = np.expand_dims(bboxes, axis=0)
                self._bboxes = bboxes

            @property
            def obj_list(self):
                return self._obj_list
            
            @obj_list.setter
            def obj_list(self, obj_list):
                self._obj_list = obj_list

            def __str__(self):
                res = ""
                for info in self._obj_list:
                    res += (info + "\n")
                return res

        tmp_image = image.copy()
        height, width = image.shape[:2]
        means = np.array([123., 117., 104.])
        image = np.expand_dims(cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)-means, self.img_size), 
            axis=0)

        classes, scores, bboxes = self.sess.run([self.classes, self.scores, self.pboxes], 
            feed_dict={self.inputX: image})
        if 0 == classes.size:
            return None

        classes, scores, bboxes = SSD.nms(classes, scores, bboxes)
        obj_list = self._draw_rectangle(tmp_image, classes, scores, bboxes)
        result = Obj_result()
        result.classes = classes
        result.scores = scores
        result.bboxes = bboxes
        result.show_image = tmp_image
        result.obj_list = obj_list

        if one_img:
            self.sess.close()

        return result


    def video_detect(self, from_camera=True, file_path=None):
        """
        detect object from camera
        Args:
            from_camera: boolean type, the video is from camera if true
            file_path: str type, the path of video file(parameter from_camera is False)
        """

        if from_camera:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise ValueError("Can't open the camera!")
            ret, frame = cap.read()
            cv2.namedWindow("Image", 2)
            while ret:
                result = self.image_detect(frame, False)
                if not result:
                    cv2.imshow("Image", frame)
                    ret, frame = cap.read()
                else:
                    cv2.imshow("Image", result.show_image)
                    ret, frame = cap.read()
                if 27 == cv2.waitKey(1):
                    break
            cv2.destroyAllWindows()

        else:
            try:
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
            except:
                raise ValueError("Can't open the file: {0}".format(file_path))
            cv2.namedWindow('Image', 2)
            while ret:
                result = self.image_detect(frame, False)
                if not result:
                    cv2.imshow("Image", frame)
                    ret, frame = cap.read()
                else:
                    cv2.imshow("Image", result.show_image)
                    ret, frame = cap.read()
                if 27 == cv2.waitKey(1):
                    break
            cv2.destroyAllWindows()

        self.sess.close()


class SSD_SOLVER(SSD):

    def __init__(self, batch_size, epoches, train_dataset, 
        valid_dataset, weight_file=None, append_name='PCB'):
        """
        initialize the Solver of SSD network
        Args:
            train_dataset: 用于训练的数据集提取对象
            valid_dataset: 用于验证模型准确率的验证集对象
            weight_file: 用于初始化的权重文件路径
            append_name: 用于保存训练得到的权重文件的文件夹的附加信息
        """

        print("Initialize parameters...")
        begin = time.time()
        SSD.__init__(self)
        if not weight_file:
            weight_file = '../weight/ssd_vgg_300_weights.ckpt'
        self.batch_size = batch_size
        self.epoches = epoches
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        # 网络信息提取
        self.inputX, self.locations, self.predictions, self.logits = SSD.build_net(self)
        # self.batch_classes = tf.placeholder(tf.int64, shape=[None, None])
        # self.batch_boxes = tf.placeholder(tf.float32, shape=[None, None, 4])

        self.batch_classes = [tf.placeholder(tf.int64, shape=[None], name='classes'+str(i)) 
            for i in range(batch_size)]
        self.batch_boxes = [tf.placeholder(tf.float32, shape=[None, 4], name='boxes'+str(i)) 
            for i in range(batch_size)]
        print(self.batch_boxes)
        self._prepare()      # 计算loss
        self.ssd_loss = tf.losses.get_total_loss()   # 得到损失函数

        # 文件处理
        self.saver = tf.train.Saver(max_to_keep=5)
        self.variable_to_restore = tf.global_variables()
        weight_dir = os.path.join('../weight', append_name)
        if not os.path.exists(weight_dir):
            os.makedirs(weight_dir) 
        self.ckpt_dir = os.path.join(weight_dir, 'ckpt')
        if not os.path.exists(self.ckpt_dir):
            os.mkdir(self.ckpt_dir)
        self.summary_dir = os.path.join(weight_dir, 'summary')
        if not os.path.exists(self.summary_dir):
            os.mkdir(self.summary_dir)

        # 模型参数保存
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.summary_dir, flush_secs=60)

        # 学习率衰减
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_step = cfg.DECAY_STEP
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_step,
            self.decay_rate, self.staircase, name='learning_rate')

        # 设置优化器
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.ssd_loss, self.global_step)
        
        # 会话创建
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        if self.sess:
            self.sess.close()
        self.sess = tf.Session(config=config)      # 创建会话
        self.sess.run(tf.global_variables_initializer())
        print("Finish initializing parameters. Time used: %.3f s"%(time.time() - begin))

        # 取出预训练的权重
        print("Restoring weights from: ", weight_file)
        self.saver.restore(self.sess, weight_file)
        print("Finish restoring weights.")

        # tensorboard 文件写入
        self.writer = tf.summary.FileWriter(self.summary_dir, flush_secs=60)
        self.writer.add_graph(self.sess.graph)


    def train(self):
        """
        执行模型的训练过程
        """
        for step in range(1, 100):
            begin = time.time()
            batch_images, batch_classes, batch_boxes = self.train_dataset.get_data()
            val_images, val_classes, val_boxes = self.valid_dataset.get_data()
            train_feed_dict = {self.inputX: batch_images}
            train_feed_dict.update(dict(zip(self.batch_classes, batch_classes)))
            train_feed_dict.update(dict(zip(self.batch_boxes, batch_boxes)))

            loss, _ = self.sess.run([self.ssd_loss, self.train_op], feed_dict=train_feed_dict)
            locations = self.sess.run(self.locations, feed_dict=train_feed_dict)

            print('loss: ', loss)
            print(locations)


    def _prepare(self):
        self.anchors = []
        self.gclasses = []
        self.glocations = []
        self.gscores = []
        for index, size in enumerate(self.feature_map_size):
            anchors_layer = self.ssd_anchor_layer(self.img_size, size, 
                                                  self.anchor_sizes[index], 
                                                  self.anchor_ratios[index], 
                                                  self.anchor_steps[index], 
                                                  self.boxes_num[index],
                                                  offset=0.5)
            self.anchors.append(anchors_layer)

        for index in range(self.batch_size):
            # 在该循环中对batch中单张图片的信息进行编码
            labels = self.batch_classes[index]
            bboxes = self.batch_boxes[index]
            gclasses_img, gscores_img, glocations_img = self.ssd_bboxes_encode(labels,\
                                                                               bboxes, 
                                                                               self.anchors, 
                                                                               self.num_classes,
                                                                               self.prior_scaling)
            # 对单张图片的编码后信息进行组合
            self.gclasses.append(gclasses_img)
            self.glocations.append(glocations_img)
            self.gscores.append(gscores_img)

        # 将 batch_size 的维度和 feat_layer 的维度进行互换
        self.gclasses = np.array(self.gclasses)
        self.glocations = np.array(self.glocations)
        self.gscores = np.array(self.gscores)
        self.gclasses = np.transpose(self.gclasses, axes=[1, 0]).tolist()
        self.glocations = np.transpose(self.glocations, axes=[1, 0]).tolist()
        self.gscores = np.transpose(self.gscores, axes=[1, 0]).tolist()

        # 将 batch_size 维进行拼接
        for i in range(len(self.feature_map_size)):
            self.gclasses[i] = tf.stack(self.gclasses[i], axis=0)
            self.glocations[i] = tf.stack(self.glocations[i], axis=0)
            self.gscores[i] = tf.stack(self.gscores[i], axis=0)

        self.ssd_losses(self.batch_size, self.logits, self.locations, self.gclasses, 
            self.glocations, self.gscores)


    @staticmethod
    def ssd_bboxes_encode_layer(labels,
                                bboxes, 
                                anchors_layer,
                                num_classes,
                                prior_scaling=[0.1, 0.1, 0.2, 0.2],
                                dtype=tf.float32):
        """
        Encode ground truth labels and bounding boxes using SSD anchors from
        one layer.
        Args:
            labels: 1-D tensor(int64), containing ground truth labels
            bboxes: Nx4 tensor(float), bbox relative coordinates(x1, y1, x2, y2)
            anchors_layer: list type, containing [cx, cy, w, h], shape of cx and cy is 
                           [map_size, map_size, 1], shape of w and h is [num_boxes_per_pixel]
            num_classes: int type, the number of classes
            no_annotation_label: int type, refers to background, 21
            match_threshold: float, threshold for positive match with ground truth bboxes
            prior_scaling: scaling for encoded coordinations
        Returns:
            (target_labels, target_locations, target_scores): Target tensor.
        """

        anchor_x, anchor_y, anchor_w, anchor_h = anchors_layer
        anchor_xmin = anchor_x - anchor_w / 2.
        anchor_ymin = anchor_y - anchor_h / 2.
        anchor_xmax = anchor_x + anchor_w * 2.
        anchor_ymax = anchor_y + anchor_h * 2.
        anchors_area = (anchor_xmax - anchor_xmin) * (anchor_ymax - anchor_ymin)   # shape of [mapsize, mapsize, num_anchors]

        # initialize tensors...
        shape = (anchor_x.shape[0], anchor_x.shape[1], anchor_w.size)
        feat_labels = tf.zeros(shape, dtype=tf.int64)
        feat_scores = tf.zeros(shape, dtype=dtype)

        feat_xmin = tf.zeros(shape, dtype=dtype)
        feat_ymin = tf.zeros(shape, dtype=dtype)
        feat_xmax = tf.zeros(shape, dtype=dtype)
        feat_ymax = tf.zeros(shape, dtype=dtype)

        def jaccard_with_anchors(bbox):
            """
            compute jaccard score between a box and all anchors of this layer
            """
            xmin = tf.maximum(anchor_xmin, bbox[0])
            ymin = tf.maximum(anchor_ymin, bbox[1])
            xmax = tf.minimum(anchor_xmax, bbox[2])
            ymax = tf.minimum(anchor_ymax, bbox[3])
            height = tf.maximum(ymax - ymin, 0.)
            width = tf.maximum(xmax - xmin, 0.)
            inter_area = height * width
            union_area = anchors_area + (bbox[3] - bbox[1]) * (bbox[2] - bbox[0]) - inter_area
            jaccard = tf.div(inter_area, union_area)
            return jaccard

        def intersection_with_anchors(bbox):
            """
            compute intersection between box and anchors
            """
            xmin = tf.maximum(anchor_xmin, bbox[0])
            ymin = tf.maximum(anchor_ymin, bbox[1])
            xmax = tf.minimum(anchor_xmax, bbox[2])
            ymax = tf.minimum(anchor_ymax, bbox[3])
            height = tf.maximum(ymax - ymin, 0.)
            width = tf.maximum(xmax - xmin, 0.)
            inter_area = height * width
            scores = tf.div(inter_area, anchors_area)
            return scores

        def condition(i, feat_labels, feat_scores, feat_xmin, 
            feat_ymin, feat_xmax, feat_ymax):
            """
            condition: check label index
            Args:
                需要包含所有在while_loop中需要更新的参数
            """
            r = tf.less(i, tf.shape(labels))   # 遍历所有的实际目标框
            return r[0]

        def body(i, feat_labels, feat_scores, feat_xmin,
            feat_ymin, feat_xmax, feat_ymax):
            """
            Body: 在每步迭代中更新变量：feat_labels, feat_scores, 
                  feat_xmin, feat_ymin, feat_xmax, feat_ymax
            只有在符合以下原则时，才进行参数的更新：
                - jaccard > 0.5
                - 一个锚框只与一个iou最大的目标框进行匹配
            """
            label = labels[i]       # 提取第i个实际框的信息，取值范围为1~20
            bbox = bboxes[i]
            jaccard = jaccard_with_anchors(bbox)

            # Mask: check threshold + scores + no annotations + num_classes.
            mask = tf.greater(jaccard, feat_scores)     # 与当前anchor的iou最大的bbox
            # mask = tf.logical_and(mask, tf.greater(jaccard, match_threshold))      # iou大于阈值，这一步在loss中计算，用于统计正例的数目
            mask = tf.logical_and(mask, feat_scores > -0.5)
            mask = tf.logical_and(mask, label < num_classes)       
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, dtype)

            # update values of parameters using mask
            feat_labels = imask * label + (1 - imask) * feat_labels
            feat_scores = tf.where(mask, jaccard, feat_scores)        # scores 即 iou
            feat_xmin = fmask * bbox[0] + (1 - fmask) * feat_xmin
            feat_ymin = fmask * bbox[1] + (1 - fmask) * feat_ymin
            feat_xmax = fmask * bbox[2] + (1 - fmask) * feat_xmax
            feat_ymax = fmask * bbox[3] + (1 - fmask) * feat_ymax

            return [i+1, feat_labels, feat_scores, feat_xmin,\
                    feat_ymin, feat_xmax, feat_ymax]

        # Main loop definition
        i = 0
        [i, feat_labels, feat_scores,
         feat_xmin, feat_ymin, 
         feat_xmax, feat_ymax] = tf.while_loop(condition, body, 
                                               [i, feat_labels, feat_scores,
                                                feat_xmin, feat_ymin,
                                                feat_xmax, feat_ymax])
        
        # transform from [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
        feat_cx = (feat_xmin + feat_xmax) / 2
        feat_cy = (feat_ymin + feat_ymax) / 2 
        feat_w = feat_xmax - feat_xmin
        feat_h = feat_ymax - feat_ymin
        
        # encode the coordinations
        feat_cx = (feat_cx - anchor_x) / anchor_w / prior_scaling[0]
        feat_cy = (feat_cy - anchor_y) / anchor_h / prior_scaling[1]
        feat_w = tf.log(feat_w / anchor_w) / prior_scaling[2]
        feat_h = tf.log(feat_h / anchor_h) / prior_scaling[3]
        feat_locations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)

        return feat_labels, feat_scores, feat_locations


    @staticmethod
    def ssd_bboxes_encode(labels, 
                          bboxes, 
                          anchors,
                          num_classes,
                          prior_scaling=[0.1, 0.1, 0.2, 0.2],
                          dtype=tf.float32, 
                          scope='ssd_bboxes_encode'):
        """
        对输入的实际标签以及实际目标框进行编码，通过ssd_bboxes_encode_layer()函数将
        实际目标框分别与每一层中的anchors进行匹配。该函数是只针对一张图片进行编码的，而
        不是一个batch
        Args:
            labels: 1-D tensor(tf.int64), 图片中每一个目标框所对应的类别
            bboxes: Nx4 tensor(float), 实际目标框的相对坐标
            anchors: 所有 feat_map 所对应的锚框列表，列表中的每一个元素都是 3-D tensor
            num_classes: int type, 类别数
            no_annotation_label:
            match_threshold: 锚框与目标框匹配的阈值
            prior_scaling: 对目标框坐标进行编码时 x, y, w, h 分别缩放的尺度
        Returns:
            (target_labels, target_scores, target_locations):
                实际的目标框分别与每一个特征图中锚框进行匹配的结果
        """
        with tf.name_scope(scope):
            target_labels = []
            target_scores = []
            target_locations = []
            for i, anchors_layer in enumerate(anchors):      # 分别对每一层特征图进行计算
                with tf.name_scope('bboxes_encode_block_%i'%i):
                    feat_labels, feat_scores, feat_locations = \
                        SSD_SOLVER.ssd_bboxes_encode_layer(labels, 
                                                           bboxes, 
                                                           anchors_layer,
                                                           num_classes,
                                                           prior_scaling, 
                                                           dtype)
                    target_labels.append(feat_labels)
                    target_scores.append(feat_scores)
                    target_locations.append(feat_locations)
            return target_labels, target_scores, target_locations


    def ssd_losses(self, batch_size, logits, locations, gclasses, 
        glocations, gscores, match_threshold=0.5, negative_ratio=3.,
        alpha=1., label_smoothing=0., device='/gpu:0', scope=None):
        """
        计算SSD网络的损失函数，包含位置误差及类别误差的加权和
        Args:
            batch_size: int type
            logits: list type, 6 层特征图中每一层的预测类别输出，列表中每一元素的shape为：
                    [batch_size, map_size, map_size, num_boxes, num_classes]，不同的层只
                    有中间的三个维度有差别
            locations: list type, 6 层特征图中每一层的预测位置输出，列表中每一元素的shape为：
                       [batch_size, map_size, map_size, num_boxes, 4]，不同的层
                       只有中间的三个维度有差别
            gclasses: list type, 一个batch中 6 层特征图中每一层的实际类别，列表中每一元素的shape为：
                      [batch_size, map_size, map_size, num_boxes]
            glocations: list type, 一个batch中 6 层特征图中每一层的anchor对应的实际目标框坐标，列表中每一
                        元素的shape为：[batch_size, map_size, map_size, num_boxes, 4]
            gscores: list type, 一个batch中 6 层特征图中每一层的anchor对应的得分(iou)，列表中每一元素的
                     shape 为：[batch_size, map_size, map_size, num_boxes]
            match_threshold: 锚框与目标框匹配的阈值
            negative_ratio: 负例数目与正例数目的比例
        """

        with tf.name_scope(scope, 'ssd_losses'):
            shape = logits[0].get_shape().as_list()
            num_classes = shape[-1]

            # flatten out all tensors
            flogits = []
            flocations = []
            fgclasses = []
            fglocations = []
            fgscores = []
            for i in range(len(logits)):     # 按特征图的层数进行遍历
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                flocations.append(tf.reshape(locations[i], [-1, 4]))
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fglocations.append(tf.reshape(glocations[i], [-1, 4]))
                fgscores.append(tf.reshape(gscores[i], [-1]))

            # concatenate the list
            logits = tf.concat(flogits, axis=0)
            locations = tf.concat(flocations, axis=0)
            gclasses = tf.concat(fgclasses, axis=0)
            glocations = tf.concat(fglocations, axis=0)
            gscores = tf.concat(fgscores, axis=0)
            dtype = logits.dtype

            # compute positive matching mask
            pmask = gscores > match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positive = tf.reduce_sum(fpmask)    # 统计正例的数目

            # hard negative mining
            # 为了保证正负样本尽量平衡，SSD采用了hard negative mining，就是对负样本进行抽样，
            # 抽样时按照置信度误差（预测背景的置信度越小，误差越大）进行降序排序，选取误差较大
            # （即置信度较小）的top-K（即置信度较小）作为训练的负样本，以保证正负样本比例接近1：3
            no_classes = tf.cast(pmask, tf.int32)
            predictions = tf.nn.softmax(logits)
            nmask = tf.logical_and(tf.logical_not(pmask),
                                   gscores > -0.5)
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask, 
                               predictions[:, 0],     # 预测的第一项为背景的概率，之后为实际物体的概率
                               1. - fnmask)
            nvalues_flat = tf.reshape(nvalues, [-1])

            # number of negative entries to select
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)     # 计算得到负例的总数
            n_neg = tf.cast(negative_ratio * n_positive, tf.int32) + batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)   # 需要通过抽样得到的负例数目

            # 抽样得到前 n_neg 个负例
            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
            max_hard_pred = -val[-1]
            # final negtive mask
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)  # 选取置信度较小的top-k
            fnmask = tf.cast(nmask, dtype)

            # calculate cross_entropy loss
            with tf.name_scope('cross_entropy_positive'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
                loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')  # 提取正例计算交叉熵
                tf.losses.add_loss(loss)

            with tf.name_scope('cross_entropy_negative'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)
                loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')  # 提取负例计算交叉熵
                tf.losses.add_loss(loss)

            # calculate location loss: smooth_l1
            def smooth_L1_loss(glocations, pred_locations):
                """
                通过smooth_L1_loss计算坐标偏差的损失函数
                Args:
                    glocations: 2-D tensor, 真实的目标框坐标
                    pred_locations: 2-D tensor, 预测得到的目标框坐标
                """
                absolute_loss = tf.abs(glocations - pred_locations)
                square_loss = 0.5 * tf.square(absolute_loss)
                l1_loss = tf.where(absolute_loss < 1, square_loss, absolute_loss - 0.5)
                return tf.reduce_sum(l1_loss, axis=-1)

            with tf.name_scope('localization'):
                weight = tf.expand_dims(alpha * fpmask, axis=-1)  # 进行维度扩展，为了后续矩阵相乘时可以广播
                loss = smooth_L1_loss(glocations, locations)
                loss = tf.div(tf.reduce_sum(loss * weight), batch_size, name='value')
                tf.losses.add_loss(loss)


if __name__ == '__main__':
    # ssd = SSD()

    # detector = SSD_DETECTOR('../weight/ssd_vgg_300_weights.ckpt')
    image = cv2.imread("../../yolo-v1/test-images/1.jpg")
    # cv2.imshow("image", image)
    # result = detector.image_detect(image)
    # print(str(result))
    # cv2.namedWindow("detect image", 2)
    # cv2.imshow("detect image", result.show_image)
    # cv2.waitKey(0)
    # detector.video_detect()
    train_dataset = Pascal_voc('train', batch_size=2)
    test_dataset = Pascal_voc('test', batch_size=2)
    image, * = train_dataset.get_data()
    solver = SSD_SOLVER(2, 15, train_dataset, test_dataset)
    solver.train()