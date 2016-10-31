# 为了 Python2 玩家们
from __future__ import print_function, division

# 第三方
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np

# 我们自己
import load

train_samples, train_labels = load._train_samples, load._train_labels
test_samples, test_labels = load._test_samples, load._test_labels

print('Training set', train_samples.shape, train_labels.shape)
print('    Test set', test_samples.shape, test_labels.shape)

image_size = load.image_size
num_labels = load.num_labels
num_channels = load.num_channels


def get_chunk(samples, labels, chunkSize):
	'''
	Iterator/Generator: get a batch of data
	这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
	用于 for loop， just like range() function
	'''
	if len(samples) != len(labels):
		raise Exception('Length of samples and labels must equal')
	stepStart = 0  # initial step
	i = 0
	while stepStart < len(samples):
		stepEnd = stepStart + chunkSize
		if stepEnd < len(samples):
			yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
			i += 1
		stepStart = stepEnd


class Network():
	def __init__(self, train_batch_size, test_batch_size, pooling_scale):
		'''
		@num_hidden: 隐藏层的节点数量
		@batch_size：因为我们要节省内存，所以分批处理数据。每一批的数据量。
		'''
		self.batch_size = train_batch_size
		self.test_batch_size = test_batch_size

		# Hyper Parameters
		self.conv_config = []		# list of dict
		self.fc_config = []			# list of dict
		self.conv_weights = []
		self.conv_biases = []
		self.fc_weights = []
		self.fc_biases = []
		self.pooling_scale = pooling_scale
		self.pooling_stride = pooling_scale

		# Graph Related
		self.graph = tf.Graph()
		self.tf_train_samples = None
		self.tf_train_labels = None
		self.tf_test_samples = None
		self.tf_test_labels = None

		# 统计
		self.merged = None
		self.train_summaries = []
		self.test_summaries = []

		# 初始化
		self.define_inputs()
		self.session = tf.Session(graph=self.graph)

		# TensorBoard Visualization
		self.writer = tf.train.SummaryWriter('./board', self.graph)

	def add_conv(self, *, patch_size, in_depth, out_depth, activation='relu', pooling=False, name):
		'''
		This function does not define operations in the graph, but only store config in self.conv_layer_config
		'''
		self.conv_config.append({
			'patch_size': patch_size,
			'in_depth': in_depth,
			'out_depth': out_depth,
			'activation': activation,
			'pooling': pooling,
			'name': name
		})
		with self.graph.as_default():
			with tf.name_scope(name):
				weights = tf.Variable(
					tf.truncated_normal([patch_size, patch_size, in_depth, out_depth], stddev=0.1))
				biases = tf.Variable(tf.constant(0.1, shape=[out_depth]))
				self.conv_weights.append(weights)
				self.conv_biases.append(biases)

	def add_fc(self, *, in_num_nodes, out_num_nodes, activation='relu', name):
		'''
		add fc layer config to slef.fc_layer_config
		'''
		self.fc_config.append({
			'in_num_nodes': in_num_nodes,
			'out_num_nodes': out_num_nodes,
			'activation': activation,
			'name': name
		})
		with self.graph.as_default():
			with tf.name_scope(name):
				weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1))
				biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes]))
				self.fc_weights.append(weights)
				self.fc_biases.append(biases)
				self.train_summaries.append(tf.histogram_summary(str(len(self.fc_weights))+'_weights', weights))
				self.train_summaries.append(tf.histogram_summary(str(len(self.fc_biases))+'_biases', biases))

	# should make the definition as an exposed API, instead of implemented in the function
	def define_inputs(self):
		with self.graph.as_default():
			# 这里只是定义图谱中的各种变量
			with tf.name_scope('inputs'):
				self.tf_train_samples = tf.placeholder(
					tf.float32, shape=(self.batch_size, image_size, image_size, num_channels), name='tf_train_samples'
				)
				self.tf_train_labels = tf.placeholder(
					tf.float32, shape=(self.batch_size, num_labels), name='tf_train_labels'
				)
				self.tf_test_samples = tf.placeholder(
					tf.float32, shape=(self.test_batch_size, image_size, image_size, num_channels), name='tf_test_samples'
				)

	def define_model(self):
		'''
		定义我的的计算图谱
		'''
		with self.graph.as_default():
			def model(data_flow, train=True):
				'''
				@data: original inputs
				@return: logits
				'''
				# Define Convolutional Layers
				for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
					with tf.name_scope(config['name'] + 'model'):
						with tf.name_scope('convolution'):
							# default 1,1,1,1 stride and SAME padding
							conv = tf.nn.conv2d(data_flow, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
							addition = conv + biases
						if config['activation'] == 'relu':
							data_flow = tf.nn.relu(addition)
						else:
							raise Exception('Activation Func can only be Relu right now. You passed', config['activation'])
						if config['pooling']:
							data_flow = tf.nn.max_pool(
								data_flow,
								ksize=[1, self.pooling_scale, self.pooling_scale, 1],
								strides=[1, self.pooling_stride, self.pooling_stride, 1],
								padding='SAME')
						# if not train:
						# 	visualize_filter_map(hidden, how_many=self.conv1_depth, name='conv1_relu')

				# Define Fully Connected Layers
				for i, (weights, biases, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):
					if i == 0:
						shape = data_flow.get_shape().as_list()
						data_flow = tf.reshape(data_flow, [shape[0], shape[1] * shape[2] * shape[3]])
					with tf.name_scope(config['name'] + 'model'):
						data_flow = tf.matmul(data_flow, weights) + biases
						if config['activation'] == 'relu':
							data_flow = tf.nn.relu(data_flow)
						elif config['activation'] is None:
							pass
						else:
							raise Exception('Activation Func can only be Relu or None right now. You passed', config['activation'])
				return data_flow

			# Training computation.
			logits = model(self.tf_train_samples)
			with tf.name_scope('loss'):
				self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.tf_train_labels))
				self.train_summaries.append(tf.scalar_summary('Loss', self.loss))

			# Optimizer.
			with tf.name_scope('optimizer'):
				self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

			# Predictions for the training, validation, and test data.
			with tf.name_scope('train'):
				self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
			with tf.name_scope('test'):
				self.test_prediction = tf.nn.softmax(model(self.tf_test_samples, train=False), name='test_prediction')

			self.merged_train_summary = tf.merge_summary(self.train_summaries)
			# self.merged_test_summary = tf.merge_summary(self.test_summaries)

	def run(self):
		'''
		用到Session
		'''
		# private function
		def print_confusion_matrix(confusionMatrix):
			print('Confusion    Matrix:')
			for i, line in enumerate(confusionMatrix):
				print(line, line[i] / np.sum(line))
			a = 0
			for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
				a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
				print(column[i] / np.sum(column), )
			print('\n', np.sum(confusionMatrix), a)

		with self.session as session:
			tf.initialize_all_variables().run()

			### 训练
			print('Start Training')
			# batch 1000
			for i, samples, labels in get_chunk(train_samples, train_labels, chunkSize=self.batch_size):
				_, l, predictions, summary = session.run(
					[self.optimizer, self.loss, self.train_prediction, self.merged_train_summary],
					feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
				)
				self.writer.add_summary(summary, i)
				# labels is True Labels
				accuracy, _ = self.accuracy(predictions, labels)
				if i % 50 == 0:
					print('Minibatch loss at step %d: %f' % (i, l))
					print('Minibatch accuracy: %.1f%%' % accuracy)
			###

			# ### 测试
			accuracies = []
			confusionMatrices = []
			for i, samples, labels in get_chunk(test_samples, test_labels, chunkSize=self.test_batch_size):
				print('samples shape', samples.shape)
				# result, summary = session.run(
				# 	[self.test_prediction, self.merged_test_summary],
				# result = session.run(
				# 	[self.test_prediction],
				# 	feed_dict={self.tf_test_samples: samples}
				# )
				result = self.test_prediction.eval(feed_dict={self.tf_test_samples: samples})
				# self.writer.add_summary(summary, i)
				accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
				accuracies.append(accuracy)
				confusionMatrices.append(cm)
				print('Test Accuracy: %.1f%%' % accuracy)
			print(' Average  Accuracy:', np.average(accuracies))
			print('Standard Deviation:', np.std(accuracies))
			print_confusion_matrix(np.add.reduce(confusionMatrices))
		###

	def accuracy(self, predictions, labels, need_confusion_matrix=False):
		'''
		计算预测的正确率与召回率
		@return: accuracy and confusionMatrix as a tuple
		'''
		_predictions = np.argmax(predictions, 1)
		_labels = np.argmax(labels, 1)
		cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
		# == is overloaded for numpy array
		accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
		return accuracy, cm

	def visualize_filter_map(self, tensor, *, how_many, name):
		filter_map = tensor[-1]
		filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
		filter_map = tf.reshape(filter_map, (how_many, 32, 32, 1))
		self.test_summaries.append(tf.image_summary(name, tensor=filter_map, max_images=1))


if __name__ == '__main__':
	net = Network(train_batch_size=64, test_batch_size=500, pooling_scale=2)

	#
	net.add_conv(patch_size=3, in_depth=1, out_depth=1, activation='relu', pooling=False, name='conv1')
	net.add_conv(patch_size=3, in_depth=1, out_depth=1, activation='relu', pooling=True, name='conv2')
	net.add_conv(patch_size=3, in_depth=1, out_depth=1, activation='relu', pooling=False, name='conv3')
	net.add_conv(patch_size=3, in_depth=1, out_depth=1, activation='relu', pooling=True, name='conv4')

	# 4 = 两次 pooling, 每一次缩小为 1/2
	# 16 = conv4 out_depth
	net.add_fc(in_num_nodes=(image_size // 4) * (image_size // 4) * 1, out_num_nodes=1, activation='relu', name='fc1')
	net.add_fc(in_num_nodes=1, out_num_nodes=10, activation=None, name='fc2')

	net.define_model()
	net.run()
