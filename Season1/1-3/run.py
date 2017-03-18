# encoding: utf-8
# 为了 Python3 的兼容，如果你用的 Python2.7
from __future__ import print_function, division
import tensorflow as tf

print('Loaded TF version', tf.__version__, '\n\n')

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
	print(addv)
	print(type(addv))
	print(type(v1))

	c1 = tf.constant(10)
	c2 = tf.constant(5)
	addc = c1 + c2
	print(addc)
	print(type(addc))
	print(type(c1))

	# 用来运行计算图谱的对象/实例？
	# session is a runtime
	sess = tf.Session()

	# Variable -> 初始化 -> 有值的Tensor
	tf.initialize_all_variables().run(session=sess)

	print('变量是需要初始化的')
	print('加法(v1, v2) = ', addv.eval(session=sess))
	print('加法(v1, v2) = ', sess.run(addv))
	print('加法(c1, c2) = ', addc.eval(session=sess))
	print('\n\n')
	#这种定义操作，再执行操作的模式被称之为“符号式编程” Symbolic Programming

	# tf.Graph.__init__()
	# Creates a new, empty Graph.
	graph = tf.Graph()
	with graph.as_default():
		value1 = tf.constant([1,2])
		value2 = tf.Variable([3,4])
		mul = value1 / value2

	with tf.Session(graph=graph) as mySess:
		tf.initialize_all_variables().run()
		print('一一对应的除法(value1, value2) = ', mySess.run(mul))
		print('一一对应的除法(value1, value2) = ', mul.eval())

	# tensor.eval(session=sess)
	# sess.run(tensor)


# 省内存？placeholder才是王道
# def use_placeholder():
	graph = tf.Graph()
	with graph.as_default():
		value1 = tf.placeholder(dtype=tf.float64)
		value2 = tf.Variable([3, 4], dtype=tf.float64)
		mul = value1 * value2

	with tf.Session(graph=graph) as mySess:
		tf.initialize_all_variables().run()
		# 我们想象一下这个数据是从远程加载进来的
		# 文件，网络
		# 假装是 10 GB
		value = load_from_remote()
		for partialValue in load_partial(value, 2):
			# runResult = mySess.run(mul, feed_dict={value1: partialValue})
			evalResult = mul.eval(feed_dict={value1: partialValue})
			print('乘法(value1, value2) = ', runResult)
		# cross validation

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

if __name__ == '__main__':
	basic_operation()
	# use_placeholder()
