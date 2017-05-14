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
    """
    Iterator/Generator: get a batch of data
    这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    用于 for loop， just like range() function
    """
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
    def __init__(self, num_hidden, batch_size, conv_depth, patch_size, pooling_scale):
        """
        @num_hidden: 隐藏层的节点数量
        @batch_size：因为我们要节省内存，所以分批处理数据。每一批的数据量。
        """
        self.batch_size = batch_size
        self.test_batch_size = 500

        # Hyper Parameters
        self.num_hidden = num_hidden
        self.patch_size = patch_size  # 滑窗的大小
        self.conv1_depth = conv_depth
        self.conv2_depth = conv_depth
        self.conv3_depth = conv_depth
        self.conv4_depth = conv_depth
        self.last_conv_depth = self.conv4_depth
        self.pooling_scale = pooling_scale
        self.pooling_stride = self.pooling_scale  # Max Pooling Stride

        # Graph Related
        self.graph = tf.Graph()
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None
        self.tf_test_prediction = None

        # 统计
        self.merged = None
        self.train_summaries = []
        self.test_summaries = []

        # 初始化
        self.define_graph()
        self.session = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter('./board', self.graph)

    def define_graph(self):
        """
        定义我的的计算图谱
        """
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

            with tf.name_scope('conv1'):
                conv1_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, num_channels, self.conv1_depth], stddev=0.1))
                conv1_biases = tf.Variable(tf.zeros([self.conv1_depth]))

            with tf.name_scope('conv2'):
                conv2_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv1_depth, self.conv2_depth],
                                        stddev=0.1))
                conv2_biases = tf.Variable(tf.constant(0.1, shape=[self.conv2_depth]))

            with tf.name_scope('conv3'):
                conv3_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv2_depth, self.conv3_depth],
                                        stddev=0.1))
                conv3_biases = tf.Variable(tf.constant(0.1, shape=[self.conv3_depth]))

            with tf.name_scope('conv4'):
                conv4_weights = tf.Variable(
                    tf.truncated_normal([self.patch_size, self.patch_size, self.conv3_depth, self.conv4_depth],
                                        stddev=0.1))
                conv4_biases = tf.Variable(tf.constant(0.1, shape=[self.conv4_depth]))

            # fully connected layer 1, fully connected
            with tf.name_scope('fc1'):
                down_scale = self.pooling_scale ** 2  # because we do 2 times pooling of stride 2
                fc1_weights = tf.Variable(
                    tf.truncated_normal(
                        [(image_size // down_scale) * (image_size // down_scale) * self.last_conv_depth,
                         self.num_hidden], stddev=0.1))
                fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.num_hidden]))

                self.train_summaries.append(tf.summary.histogram('fc1_weights', fc1_weights))
                self.train_summaries.append(tf.summary.histogram('fc1_biases', fc1_biases))

            # fully connected layer 2 --> output layer
            with tf.name_scope('fc2'):
                fc2_weights = tf.Variable(tf.truncated_normal([self.num_hidden, num_labels], stddev=0.1),
                                          name='fc2_weights')
                fc2_biases = tf.Variable(tf.constant(0.1, shape=[num_labels]), name='fc2_biases')
                self.train_summaries.append(tf.summary.histogram('fc2_weights', fc2_weights))
                self.train_summaries.append(tf.summary.histogram('fc2_biases', fc2_biases))

            # 想在来定义图谱的运算
            def model(data, train=True):
                """
                @data: original inputs
                @return: logits
                """
                with tf.name_scope('conv1_model'):
                    with tf.name_scope('convolution'):
                        conv1 = tf.nn.conv2d(data, filter=conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
                        addition = conv1 + conv1_biases
                    hidden = tf.nn.relu(addition)

                    if not train:
                        # transpose the output of an activation to image
                        # conv1_activation_relu shape: (8, 32, 32, 64)
                        # 64 filter maps from this convolution, that's 64 grayscale images
                        # image size is 32x32
                        # 8 is the batch_size, which means 8 times of convolution was performed
                        # just use the last one (index 7) as record

                        filter_map = hidden[-1]
                        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
                        filter_map = tf.reshape(filter_map, (self.conv1_depth, 32, 32, 1))
                        self.test_summaries.append(
                            tf.summary.image('conv1_relu', tensor=filter_map, max_outputs=self.conv1_depth))

                with tf.name_scope('conv2_model'):
                    with tf.name_scope('convolution'):
                        conv2 = tf.nn.conv2d(hidden, filter=conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
                        addition = conv2 + conv2_biases
                    hidden = tf.nn.relu(addition)
                    hidden = tf.nn.max_pool(
                        hidden,
                        ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                        strides=[1, self.pooling_stride, self.pooling_stride, 1],
                        padding='SAME')

                with tf.name_scope('conv3_model'):
                    with tf.name_scope('convolution'):
                        conv3 = tf.nn.conv2d(hidden, filter=conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
                        addition = conv3 + conv3_biases
                    hidden = tf.nn.relu(addition)

                with tf.name_scope('conv4_model'):
                    with tf.name_scope('convolution'):
                        conv4 = tf.nn.conv2d(hidden, filter=conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
                        addition = conv4 + conv4_biases
                    hidden = tf.nn.relu(addition)
                    # if not train:
                    # 	filter_map = hidden[-1]
                    # 	filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
                    # 	filter_map = tf.reshape(filter_map, (self.conv4_depth, 16, 16, 1))
                    # 	tf.image_summary('conv4_relu', tensor=filter_map, max_images=self.conv4_depth)
                    hidden = tf.nn.max_pool(
                        hidden,
                        ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                        strides=[1, self.pooling_stride, self.pooling_stride, 1],
                        padding='SAME')

                # fully connected layer 1
                shape = hidden.get_shape().as_list()
                reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

                with tf.name_scope('fc1_model'):
                    fc1_model = tf.matmul(reshape, fc1_weights) + fc1_biases
                    hidden = tf.nn.relu(fc1_model)

                # fully connected layer 2
                with tf.name_scope('fc2_model'):
                    return tf.matmul(hidden, fc2_weights) + fc2_biases

            # Training computation.
            logits = model(self.tf_train_samples)
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.tf_train_labels))
                self.train_summaries.append(tf.summary.scalar('Loss', self.loss))

            # Optimizer.
            with tf.name_scope('optimizer'):
                self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            with tf.name_scope('train'):
                self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
            with tf.name_scope('test'):
                self.test_prediction = tf.nn.softmax(model(self.tf_test_samples, train=False), name='test_prediction')

            self.merged_train_summary = tf.summary.merge_all()
            self.merged_test_summary = tf.summary.merge_all()

    def run(self):
        """
        用到Session
        """

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

            ### 测试
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in get_chunk(test_samples, test_labels, chunkSize=self.test_batch_size):
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples}
                )
                # result = self.test_prediction.eval()
                self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            print_confusion_matrix(np.add.reduce(confusionMatrices))
        ###

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        """
        计算预测的正确率与召回率
        @return: accuracy and confusionMatrix as a tuple
        """
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        # == is overloaded for numpy array
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm


if __name__ == '__main__':
    net = Network(num_hidden=16, batch_size=64, patch_size=3, conv_depth=16, pooling_scale=2)
    net.run()
