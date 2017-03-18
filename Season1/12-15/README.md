<<<<<<< HEAD
关于卷积神经网络的理论知识，请一定阅读 [cs231n 的课件](http://cs231n.github.io/convolutional-networks/)。
虽然是英文的，但是内容浅显易读，又不失细节与深度，是理解卷积神经网络的绝佳资料。

[Theano 的教程也有很详细的介绍 + 很爽的动画效果](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html)。虽然这里是TF的教程，但是知识是互通的。

本系列是“编程向”，所以一切理论都是点到为止。然后外链更优质的理论资源。这样大家的学习效率才高。

### 第 13 期涉及到的新概念
#### Max Pooling
Pooling 是图片的缩小（Downscaling）。这个操作是损失精度的。假如说 Pooling 的 scale 是 2。那么也就是将图片长宽各缩小至 1/2。也就是每 4 个像素点只取一个。  
那么，Max Pooling 则是取最大值的那一个像素。而 Average Pooling 就是取 4 个像素点的平均值。据研究表明 Max Pooling 通常效果更好，所以在代码实例中被使用。

#### Relu Layer 的含义
Relu 是激活函数，定义为： relu(x) = max(x, 0)  
或者可以写成  
relu(x) = x if x > 0 else 0    
所以，relu 就是一个线性的阀值函数而已

#### 请参考
[cs231n 关于 Convolutional Layer 架构的解释](http://cs231n.github.io/convolutional-networks/)    
[维基百科关于 Relu 的解释](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))  
=======
关于卷积神经网络的理论知识，请一定阅读 [cs231n 的课件](http://cs231n.github.io/convolutional-networks/)。
虽然是英文的，但是内容浅显易读，又不失细节与深度，是理解卷积神经网络的绝佳资料。

[Theano 的教程也有很详细的介绍 + 很爽的动画效果](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html)。虽然这里是TF的教程，但是知识是互通的。

本系列是“编程向”，所以一切理论都是点到为止。然后外链更优质的理论资源。这样大家的学习效率才高。

### 第 13 期涉及到的新概念
#### Max Pooling
Pooling 是图片的缩小（Downscaling）。这个操作是损失精度的。假如说 Pooling 的 scale 是 2。那么也就是将图片长宽各缩小至 1/2。也就是每 4 个像素点只取一个。  
那么，Max Pooling 则是取最大值的那一个像素。而 Average Pooling 就是取 4 个像素点的平均值。据研究表明 Max Pooling 通常效果更好，所以在代码实例中被使用。

#### Relu Layer 的含义
Relu 是激活函数，定义为： relu(x) = max(x, 0)  
或者可以写成  
relu(x) = x if x > 0 else 0    
所以，relu 就是一个线性的阀值函数而已

#### 请参考
[cs231n 关于 Convolutional Layer 架构的解释](http://cs231n.github.io/convolutional-networks/)    
[维基百科关于 Relu 的解释](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))  

## API
```Python
from dp_refined_api import Network

# 首先，通过某种方式得到你的数据
# First，get your data somehow
train_samples, train_labels, test_samples, test_labels = get_your_data()

net = Network(train_batch_size=64, test_batch_size=500, pooling_scale=2)
net.define_inputs(
  train_samples_shape=(64, image_size, image_size, num_channels),
	train_labels_shape=(64, num_labels),
	test_samples_shape=(500, image_size, image_size, num_channels)
)

net.add_conv(patch_size=3, in_depth=num_channels, out_depth=16, activation='relu', pooling=False, name='conv1')
net.add_conv(patch_size=3, in_depth=16, out_depth=16, activation='relu', pooling=True, name='conv2')

# 2 = 1次 pooling, 每一次缩小为 1/2
image_size = 32
net.add_fc(in_num_nodes=(image_size // 2) * (image_size // 2) * 16, out_num_nodes=16, activation='relu', name='fc1')
net.add_fc(in_num_nodes=16, out_num_nodes=10, activation=None, name='fc2')

# 在添加了所有层之后，定义模型
# After adding all layers, define the model
net.define_model()

# 运行网络
# Run the network
# data_iterator 是一个自定义的 Generator 函数, 用来给网络喂数据
net.run(data_iterator, train_samples, train_labels, test_samples, test_labels)
```
>>>>>>> cc8b03fe76bc6a6eedea52e7d3acd66edce665f7
