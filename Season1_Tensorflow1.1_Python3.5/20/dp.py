# 新的 refined api 不支持 Python2
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np


class Network():
    def __init__(self, train_batch_size, test_batch_size, pooling_scale,
                 dropout_rate, base_learning_rate, decay_rate,
                 optimizeMethod='adam', save_path='model/default.ckpt'):
        """
        @num_hidden: 隐藏层的节点数量
        @batch_size：因为我们要节省内存，所以分批处理数据。每一批的数据量。
        """
        self.optimizeMethod = optimizeMethod
        self.dropout_rate = dropout_rate
        self.base_learning_rate = base_learning_rate
        self.decay_rate = decay_rate

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Hyper Parameters
        self.conv_config = []  # list of dict
        self.fc_config = []  # list of dict
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weights = []
        self.fc_biases = []
        self.pooling_scale = pooling_scale
        self.pooling_stride = pooling_scale

        # Graph Related
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None

        # 统计
        self.writer = None
        self.merged = None
        self.train_summaries = []
        self.test_summaries = []

        # save 保存训练的模型
        self.saver = None
        self.save_path = save_path

    def add_conv(self, *, patch_size, in_depth, out_depth, activation='relu', pooling=False, name):
        """
        This function does not define operations in the graph, but only store config in self.conv_layer_config
        """
        self.conv_config.append({
            'patch_size': patch_size,
            'in_depth': in_depth,
            'out_depth': out_depth,
            'activation': activation,
            'pooling': pooling,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(
                tf.truncated_normal([patch_size, patch_size, in_depth, out_depth], stddev=0.1), name=name + '_weights')
            biases = tf.Variable(tf.constant(0.1, shape=[out_depth]), name=name + '_biases')
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)

    def add_fc(self, *, in_num_nodes, out_num_nodes, activation='relu', name):
        """
        add fc layer config to slef.fc_layer_config
        """
        self.fc_config.append({
            'in_num_nodes': in_num_nodes,
            'out_num_nodes': out_num_nodes,
            'activation': activation,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes]))
            self.fc_weights.append(weights)
            self.fc_biases.append(biases)
            self.train_summaries.append(tf.summary.histogram(str(len(self.fc_weights)) + '_weights', weights))
            self.train_summaries.append(tf.summary.histogram(str(len(self.fc_biases)) + '_biases', biases))

    def apply_regularization(self, _lambda):
        # L2 regularization for the fully connected parameters
        regularization = 0.0
        for weights, biases in zip(self.fc_weights, self.fc_biases):
            regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
        # 1e5
        return _lambda * regularization

    # should make the definition as an exposed API, instead of implemented in the function
    def define_inputs(self, *, train_samples_shape, train_labels_shape, test_samples_shape):
        # 这里只是定义图谱中的各种变量
        with tf.name_scope('inputs'):
            self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
            self.tf_train_labels = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_test_samples = tf.placeholder(tf.float32, shape=test_samples_shape, name='tf_test_samples')

    def define_model(self):
        """
        定义我的的计算图谱
        """

        def model(data_flow, train=True):
            """
            @data: original inputs
            @return: logits
            """
            # Define Convolutional Layers
            for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
                with tf.name_scope(config['name'] + '_model'):
                    with tf.name_scope('convolution'):
                        # default 1,1,1,1 stride and SAME padding
                        data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
                        data_flow = data_flow + biases
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                      display_size=32 // (i // 2 + 1), name=config['name'] + '_conv')
                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                      display_size=32 // (i // 2 + 1), name=config['name'] + '_relu')
                    else:
                        raise Exception('Activation Func can only be Relu right now. You passed', config['activation'])
                    if config['pooling']:
                        data_flow = tf.nn.max_pool(
                            data_flow,
                            ksize=[1, self.pooling_scale, self.pooling_scale, 1],
                            strides=[1, self.pooling_stride, self.pooling_stride, 1],
                            padding='SAME')
                        if not train:
                            self.visualize_filter_map(data_flow, how_many=config['out_depth'],
                                                      display_size=32 // (i // 2 + 1) // 2,
                                                      name=config['name'] + '_pooling')

            # Define Fully Connected Layers
            for i, (weights, biases, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):
                if i == 0:
                    shape = data_flow.get_shape().as_list()
                    data_flow = tf.reshape(data_flow, [shape[0], shape[1] * shape[2] * shape[3]])
                with tf.name_scope(config['name'] + 'model'):

                    ### Dropout
                    if train and i == len(self.fc_weights) - 1:
                        data_flow = tf.nn.dropout(data_flow, self.dropout_rate, seed=4926)
                    ###

                    data_flow = tf.matmul(data_flow, weights) + biases
                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                    elif config['activation'] is None:
                        pass
                    else:
                        raise Exception('Activation Func can only be Relu or None right now. You passed',
                                        config['activation'])
            return data_flow

        # Training computation.
        logits = model(self.tf_train_samples)
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.tf_train_labels))
            self.loss += self.apply_regularization(_lambda=5e-4)
            self.train_summaries.append(tf.summary.scalar('Loss', self.loss))

        # learning rate decay
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.base_learning_rate,
            global_step=global_step * self.train_batch_size,
            decay_steps=100,
            decay_rate=self.decay_rate,
            staircase=True
        )

        # Optimizer.
        with tf.name_scope('optimizer'):
            if (self.optimizeMethod == 'gradient'):
                self.optimizer = tf.train \
                    .GradientDescentOptimizer(learning_rate) \
                    .minimize(self.loss)
            elif (self.optimizeMethod == 'momentum'):
                self.optimizer = tf.train \
                    .MomentumOptimizer(learning_rate, 0.5) \
                    .minimize(self.loss)
            elif (self.optimizeMethod == 'adam'):
                self.optimizer = tf.train \
                    .AdamOptimizer(learning_rate) \
                    .minimize(self.loss)

        # Predictions for the training, validation, and test data.
        with tf.name_scope('train'):
            self.train_prediction = tf.nn.softmax(logits, name='train_prediction')
            tf.add_to_collection("prediction", self.train_prediction)
        with tf.name_scope('test'):
            self.test_prediction = tf.nn.softmax(model(self.tf_test_samples, train=False), name='test_prediction')
            tf.add_to_collection("prediction", self.test_prediction)

            single_shape = (1, 32, 32, 1)
            single_input = tf.placeholder(tf.float32, shape=single_shape, name='single_input')
            self.single_prediction = tf.nn.softmax(model(single_input, train=False), name='single_prediction')
            tf.add_to_collection("prediction", self.single_prediction)

        self.merged_train_summary = tf.summary.merge(self.train_summaries)
        self.merged_test_summary = tf.summary.merge(self.test_summaries)

        # 放在定义Graph之后，保存这张计算图
        self.saver = tf.train.Saver(tf.all_variables())

    def run(self, train_samples, train_labels, test_samples, test_labels, *, train_data_iterator, iteration_steps,
            test_data_iterator):
        """
        用到Session
        :data_iterator: a function that yields chuck of data
        """
        self.writer = tf.summary.FileWriter('./board', tf.get_default_graph())

        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.initialize_all_variables().run()

            ### 训练
            print('Start Training')
            # batch 1000
            for i, samples, labels in train_data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                          chunkSize=self.train_batch_size):
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
            for i, samples, labels in test_data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                result, summary = session.run(
                    [self.test_prediction, self.merged_test_summary],
                    feed_dict={self.tf_test_samples: samples}
                )
                self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.print_confusion_matrix(np.add.reduce(confusionMatrices))
            ###

    def train(self, train_samples, train_labels, *, data_iterator, iteration_steps):
        self.writer = tf.summary.FileWriter('./board', tf.get_default_graph())
        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.initialize_all_variables().run()

            ### 训练
            print('Start Training')
            # batch 1000
            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                    chunkSize=self.train_batch_size):
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

            # 检查要存放的路径值否存在。这里假定只有一层路径。
            import os
            if os.path.isdir(self.save_path.split('/')[0]):
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)
            else:
                os.makedirs(self.save_path.split('/')[0])
                save_path = self.saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)

    def test(self, test_samples, test_labels, *, data_iterator):
        if self.saver is None:
            self.define_model()
        if self.writer is None:
            self.writer = tf.summary.FileWriter('./board', tf.get_default_graph())

        print('Before session')
        with tf.Session(graph=tf.get_default_graph()) as session:
            self.saver.restore(session, self.save_path)
            ### 测试
            accuracies = []
            confusionMatrices = []
            for i, samples, labels in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                result = session.run(
                    self.test_prediction,
                    feed_dict={self.tf_test_samples: samples}
                )
                # self.writer.add_summary(summary, i)
                accuracy, cm = self.accuracy(result, labels, need_confusion_matrix=True)
                accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            print(' Average  Accuracy:', np.average(accuracies))
            print('Standard Deviation:', np.std(accuracies))
            self.print_confusion_matrix(np.add.reduce(confusionMatrices))
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

    def visualize_filter_map(self, tensor, *, how_many, display_size, name):
        # print(tensor.get_shape)
        filter_map = tensor[-1]
        # print(filter_map.get_shape())
        filter_map = tf.transpose(filter_map, perm=[2, 0, 1])
        # print(filter_map.get_shape())
        filter_map = tf.reshape(filter_map, (how_many, display_size, display_size, 1))
        # print(how_many)
        self.test_summaries.append(tf.summary.image(name, tensor=filter_map, max_outputs=how_many))

    def print_confusion_matrix(self, confusionMatrix):
        print('Confusion    Matrix:')
        for i, line in enumerate(confusionMatrix):
            print(line, line[i] / np.sum(line))
        a = 0
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
            print(column[i] / np.sum(column), )
        print('\n', np.sum(confusionMatrix), a)
