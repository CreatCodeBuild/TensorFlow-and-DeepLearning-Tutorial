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
