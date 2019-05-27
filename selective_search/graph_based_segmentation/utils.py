#-*- coding: utf-8 -*-
"""
Created on 26 March, 2019

Author: xhj
"""
import numpy as np


__all__ = ['Node', 'Forest', 'create_edge', 'build_graph', 'segment_graph',
		   'generate_image']


class Node:

	def __init__(self, parent, rank = 0, size = 1):

		if not isinstance(rank, int):
			raise ValueError('The value of rank must be int type')
		
		self.parent = parent
		self._rank = rank
		self.size = size

	def __repr__(self):
		
		info = "parent: {0} | rank: {1} | size: {2}"
		return info.format(self.parent, self._rank, self.size)

	@property
	def rank(self):
		return self._rank

	@rank.setter
	def rank(self, value):
		if not isinstance(value, int):
			raise ValueError('The value of rank must int type')
		self._rank = value


class Forest:
	"""
	class 类型，用于存储从一幅图片中分割出的所有区域吗，初始时每个像素
	都是一个单独的区域，因此在初始化时只需要输入节点的数目
	"""
	
	def __init__(self, num_nodes):

		# initialize nodes list, the parent of every nodes is it self
		self.nodes = [Node(i) for i in range(num_nodes)] 
		# initialize the region number of this graph. Every nodes is
		# a single region
		self.num_regions = num_nodes

	def sizeof_node(self, i):
		return self.nodes[i].size 

	def find_parent_of_region(self, n):
		"""
		find the parent of the region node n belongs to. The parent's
		parent node is it self

		Parameters:
		n: the id of this node

		Returns:
		parent: the id of this region's parent node
		"""

		parent = n 
		while parent != self.nodes[parent].parent:
			parent = self.nodes[parent].parent

		return parent

	def merge_region(self, a, b):
		"""
		merge the regions of node a and node b

		Parameters:
		a: the parent node of region a
		b: the parent node of region b
		"""

		# nodes a and b must be parent node
		assert((self.nodes[a].parent == a) & (self.nodes[b].parent == b))

		if self.nodes[a].rank >= self.nodes[b].rank:  # 将优先级高的节点作为合并后区域的父节点
			self.nodes[b].parent = a    # update the parent of node b
			self.nodes[a].size += self.nodes[b].size  # update the size of node a
		else:
			self.nodes[a].parent = b 
			self.nodes[b].size += self.nodes[a].size 

		self.num_regions -= 1  # the number of regions - 1

	def show_nodes(self):
		for node in self.nodes:
			print(node)

	def __repr__(self):
		info = "This forest has {0} regions and {1} nodes"
		info.format(self.num_regions, len(self.nodes))


def create_edge(img, width, x1, y1, x2, y2, diff):
	"""
	create the edge between pixel(y1, x1) and pixel(y2, x2), the weight
	of this edge is determinded by function diff. the edge has no direction

	Parameters:
	img: the image which would be segmented
	width: the width of image
	x1, y1: location of pixel1
	x2, y2: location of pixel2
	diff: the function used to calculate weight of this edge

	Returns:
	edge: type of three members tuple. (id_of_node1, id_of_node_2, weight)
	"""

	vertex_id = lambda x, y: y * width + x  # use one dimension index as id
	w = diff(img, x1, y1, x2, y2)        # calculate the weight
	edge = (vertex_id(x1, y1), vertex_id(x2, y2), w)
	return edge


def build_graph(img, diff, neighborhood_8 = False):
	"""
	create the graph based on image and diff function, the graph was stored 
	in a list use the form of (node1, node2, weight)

	Parameters:
	img: type of numpy.ndarray the input image
	diff: the function which was used to calculate weight of edges
	neighborbood_8: type of boolean, indicates use 8 neighberbood or not

	Returns:
	graph: type of list, contains the edge of graph, the edge was generated
		   by function create_edge()
	"""

	if not isinstance(img, np.ndarray):
		raise ValueError('the data type of input image must be numpy.ndarray')
	
	height, width = img.shape[:2]
	graph = set()
	for y in range(height):
		for x in range(width):
			if x > 0:
				graph.add(create_edge(img, width, x, y, x-1, y, diff))
			if y > 0:
				graph.add(create_edge(img, width, x, y, x, y-1, diff))
			if True == neighborhood_8:
				if x > 0 and y > 0:
					graph.add(create_edge(img, width, x, y, x-1, y-1, diff))
				if x > 0 and y < width - 1:
					graph.add(create_edge(img, width, x, y, x-1, y+1, diff))

	return list(graph)


def segment_graph(graph, num_nodes, k, min_size):
	"""
	segment the graph by merge similarity regions

	Parameters:
	graph: type of list, contains the edge of graph, every element is
		   (node1, node2, w)
	num_nodes: the number of initialize nodes in image
	k: the parameter to contral the internal difference
	min_size: the minimize number of nodes in every regions

	Returns:
	forest: type of class Forest, the forest after segmentaion
	"""

	weight = lambda edge: edge[2]    # lambda function which returns the weights of edges

	def remove_small_regions(graph, forest, min_size):
		"""
		merge the neighber regions which was smaller than mini_size

		Parameters:
		graph: type of list, the graph after segmentation
		forest: type of class Forest
		min_size: type of int, the minimize number of pixels that a segmentated region
				  should contains
		"""

		for edge in graph:
			parent_a = forest.find_parent_of_region(edge[0])
			parent_b = forest.find_parent_of_region(edge[1])

			if parent_a != parent_b and \
				(forest.sizeof_node(parent_a) < min_size or \
				 forest.sizeof_node(parent_b) < min_size):
				forest.merge_region(parent_a, parent_b)

	forest = Forest(num_nodes)
	sorted_graph = sorted(graph, key = weight) # sort the node pairs by their weight
	max_internal_diff = [k] * num_nodes  # initialize the max internal difference of every regins
	
	for edge in sorted_graph:
		# find the parent nodes of both nodes connected by this edge
		parent_a = forest.find_parent_of_region(edge[0])
		parent_b = forest.find_parent_of_region(edge[1])

		# compare the difference in nodes and difference between nodes
		a_condition = weight(edge) <= max_internal_diff[parent_a]
		b_condition = weight(edge) <= max_internal_diff[parent_b]

		if parent_a != parent_b and a_condition and b_condition:
			"""
			if the tree generated by node parent_a is not the same as 
			the tree generated by node parent_b and the difference 
			between this two regions is smaller than the differcence
			in this two regions, we should merge this two regions
			"""

			forest.merge_region(parent_a, parent_b)
			# find the new parent of merged region
			new_parent = forest.find_parent_of_region(parent_a)
			# update the max internal difference of the region merged just now
			max_internal_diff[new_parent] = weight(edge) + k / forest.sizeof_node(new_parent)

	remove_small_regions(graph, forest, min_size)
	return forest


def generate_image(forest, width, height):
	"""
	generate the image from forest after segmentation, the size of 
	image is (height, width) 

	Parameters:
	forest: type of class Forest
	width: the width of image
	height: the height of image

	Returns:
	segmented_img: type of numpy.ndarray, the segmented image
	"""

	random_color = lambda: np.random.randint(0, 255, 3)
	colors = [random_color() for i in range(width * height)]

	segmented_img = np.zeros((height, width, 3))
	for y in range(height):
		for x in range(width):
			index = y * width + x
			component = forest.find_parent_of_region(index)
			segmented_img[y, x, :] = colors[component][:]
			# print(colors[component][:])

	return segmented_img