# encoding: utf-8
# 为了 Python3 的兼容，如果你用的 Python2.7
from __future__ import print_function, division
import tensorflow as tf

print('Loaded TF version', tf.__version__)

# Tensor 在数学中是“张量”
# 标量，矢量/向量，张量

# 简单地理解
# 标量表示值
# 矢量表示位置（空间中的一个点）
# 张量表示整个空间

# 一维数组是矢量
# 多维数组是张量, 矩阵也是张量


# 4个重要的类型
# @Variable		计算图谱中的变量
# @Tensor		一个多维矩阵，带有很多方法
# @Graph		一个计算图谱
# @Session		用来运行一个计算图谱


# 三个重要的函数

# Variable 变量
# tf.Variable.__init__(
#	initial_value=None, @Tensor
#	trainable=True,
#	collections=None,
#	validate_shape=True,
#	caching_device=None,
#	name=None,
#	variable_def=None,
#	dtype=None)
# 注意：Variable是一个Class，Tensor也是一个Class

# Constant 常数
# tf.constant(value, dtype=None, shape=None, name='Const')
# return: a constant @Tensor

# Placeholder 暂时变量？
# tf.placeholder(dtype, shape=None, name=None)
# return: 一个还尚未存在的 @Tensor



# 让我们用计算图谱来实现一些简单的函数
# + - * / 四则运算
def basic_operation():
	v1 = tf.Variable(10)
	v2 = tf.Variable(5)
	addv = v1 + v2


	c1 = tf.constant(10)
	c2 = tf.constant(5)
	addc = c1 + c2

	# print(type(v1))
	# print(type(c1))
	# print(type(addv))

	# 用来运行计算图谱的对象/实例？
	# session is a runtime
	sess = tf.Session()

	# Variable -> 初始化 -> Tensor
	#
	tf.initialize_all_variables().run(session=sess)

	# print('变量是需要初始化的')
	# print('加法(v1, v2) = ', addv.eval(session=sess))
	# print('加法(c1, c2) = ', addc.eval(session=sess))

	# tf.Graph.__init__()
	# Creates a new, empty Graph.
	graph = tf.Graph()
	with graph.as_default():
		value1 = tf.constant([1,2])
		value2 = tf.Variable([3,4])
		mul = value1 * value2

	with tf.Session(graph=graph) as mySess:
		tf.initialize_all_variables().run()
		print('乘法(value1, value2) = ', mySess.run(mul))

	# tensor.eval(session=sess)
	# sess.run(tensor)


# 省内存？placeholder才是王道
def use_placeholder():
	graph = tf.Graph()
	with graph.as_default():
		value1 = tf.placeholder(dtype=tf.int64)
		value2 = tf.Variable([3, 4], dtype=tf.int64)
		mul = value1 * value2

	with tf.Session(graph=graph) as mySess:
		tf.initialize_all_variables().run()
		# 我们想象一下这个数据是从远程加载进来的
		# 文件，网络，设备
		# 假装是 10万 GB
		value = load_from_remote()
		for partialValue in load_partial(value, 2):
			holderValue = {value1: partialValue}
			# runResult = mySess.run(mul, feed_dict=holderValue)
			evalResult = mul.eval(feed_dict=holderValue)
			print('乘法(value1, value2) = ', evalResult)

def load_from_remote():
	return [-x for x in range(1000)]


# 自定义的 Iterator
# yield， generator function
def load_partial(value, step):
	index = 0
	while index < len(value):
		yield value[index:index+step]
		index += step
	return

# 给大家解决一些命名空间、作用域上的迷惑
def scope_and_namespace():
	#首先我们要知道的是，with语句是不会制造自己的作用域的
	# graph = tf.Graph()
	# with graph.as_default():
	# 	a = 2
	# 	value1 = tf.placeholder(dtype=tf.int64)
	# 	value2 = tf.Variable([3, 4], dtype=tf.int64)
	# 	mul = value1 * value2
	#
	# print('我们仍然可以摄取 a，a=', a)
	#
	# with tf.Session(graph=graph) as mySess:
	# 	tf.initialize_all_variables().run()
	# 	value = load_from_remote()
	# 	for partialValue in load_partial(value, 2):
	# 		# 这也是为什么我们可以在Session的with中摄取value1这个命名
	# 		holderValue = {value1: partialValue}
	# 		evalResult = mul.eval(feed_dict=holderValue)
	# 		print('乘法(value1, value2) = ', evalResult)

	# 这有什么问题呢？
	# with 下创建的Python变量（不要与TF变量混淆）其实是全部暴露给了上级作用域的
	# 如果你在全局写，那么就全部是全局变量了
	# 而且，如果你需要更好的模组化呢？
	class Network():
		def __init__(self, v3):
			self.graph = None
			self.holder = None
			self.v3 = v3
			self.operation = None

		def define_graph(self):
			self.graph = tf.Graph()
			with self.graph.as_default():
				a = 2
				self.holder = tf.placeholder(dtype=tf.int64)
				value2 = tf.Variable([3, 4], dtype=tf.int64)
				value3 = tf.Variable(self.v3, dtype=tf.int64)
				self.operation = self.holder * value2 * value3

			print('我们仍然可以摄取 a，a=', a)

		def run_session(self):
			with tf.Session(graph=self.graph) as mySess:
				tf.initialize_all_variables().run()
				value = load_from_remote()
				for partialValue in load_partial(value, 2):
					# 这也是为什么我们可以在Session的with中摄取value1这个命名
					holderValue = {self.holder: partialValue}
					evalResult = self.operation.eval(feed_dict=holderValue)
					print('乘法(value1, value2) = ', evalResult)

	# 声明式的API
	myNet = Network([0, 1])
	myNet.define_graph()
	myNet.run_session()

if __name__ == '__main__':
	# basic_operation()
	# use_placeholder()
	scope_and_namespace()
