from __future__ import print_function, division
import tensorflow as tf
import numpy as np
import load

train_dataset, train_labels = load._train_dataset, load._train_labels
test_dataset, test_labels = load._test_dataset,  load._test_labels

print('Training set', train_dataset.shape, train_labels.shape)
print('    Test set', test_dataset.shape, test_labels.shape)

image_size = load.image_size
num_labels = load.num_labels
num_channels = load.num_channels

class Network():
	def __init__(self, num_hidden, batch_size):
		'''
		@num_hidden: 隐藏层的节点数量
		@batch_size：因为我们要节省内存，所以分批处理数据。每一批的数据量。
		'''
		self.batch_size = batch_size

		# Hyper Parameters
		self.num_hidden = num_hidden

		# Graph Related
		self.graph = tf.Graph()
		self.tf_train_samples = None
		self.tf_train_labels = None
		self.tf_test_samples = None
		self.tf_test_labels = None
		self.tf_test_prediction = None

	def define_graph(self):
		'''
		定义我的的计算图谱
		'''
		with self.graph.as_default():
			# 这里只是定义图谱中的各种变量
			self.tf_train_samples = tf.placeholder(
				tf.float32, shape=(self.batch_size, image_size, image_size, num_channels)
			)
			self.tf_train_labels  = tf.placeholder(
				tf.float32, shape=(self.batch_size, num_labels)
			)
			self.tf_test_samples  = tf.placeholder(
				tf.float32, shape=(self.testing_batch_size, image_size, image_size, num_channels)
			)

			# fully connected layer 1, fully connected
			fc1_weights = tf.Variable(
				tf.truncated_normal([image_size * image_size * self.batch_size, self.num_hidden], stddev=0.1)
			)
			fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]))

			# fully connected layer 2 --> output layer
			fc2_weights = tf.Variable(
				tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1)
			)
			fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]))

			# 想在来定义图谱的运算
			def model(data):
				# fully connected layer 1
				shape = data.get_shape().as_list()
				reshape = tf.reshape(data, [shape[0], shape[1] * shape[2] * shape[3]])
				hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

				# fully connected layer 2
				return tf.matmul(hidden, fc2_weights) + fc2_biases

			# Training computation.
			logits = model(self.tf_train_samples)
			loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels)
			)

			# Optimizer.
			optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

			# Predictions for the training, validation, and test data.
			train_prediction = tf.nn.softmax(logits)
			test_prediction = tf.nn.softmax(model(self.tf_test_samples))

	def train(self):
		'''
		用到Session
		'''

	def test(self):
		'''
		用到Session
		'''

	def accuracy(self):
		'''
		'''
