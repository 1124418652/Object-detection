# -*- coding: utf-8 -*-
"""
Created on 2019/5/26

@Author: xhj
"""

import time 
import datetime


class Timer(object):
	"""
	A simple timer class
	"""

	def __init__(self):
		self.init_time = time.time()
		self.total_time = 0.       
		self.calls = 0           # 记录函数调用次数
		self.start_time = 0.
		self.diff = 0.
		self.average_time = 0.
		self.remain_time = 0.

	def tic(self):
		"""
		using time.time instead of time.clock because time time.clock
		does not normalize for multithreading
		"""
		self.start_time = time.time()

	def toc(self, average=True):
		"""
		calculate the time used for from call tic func to now
		"""
		self.diff = time.time() - self.start_time
		self.total_time += self.diff 
		self.calls += 1
		if average:
			return self.total_time / self.calls
		else:
			return self.diff

	def remain(self, iters, max_iters):
		if 0 == iters:
			self.remain_time = 0
		else:
			self.remain_time = (time.time() - self.init_time) * \
				(max_iters - iters) / iters
		return str(datetime.timedelta(seconds=int(self.remain_time)))


if __name__ == '__main__':
	timer = Timer()
	print(timer.remain(10, 10))