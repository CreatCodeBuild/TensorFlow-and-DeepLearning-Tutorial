## 这一次给大家讲一讲一些常用的优化的方法。
### Regularization
L1 与 L2 Loss Function 的一些问题。给FC用的。但是我实际上不太懂。[Quora答案](https://www.quora.com/What-is-the-difference-between-L1-and-L2-regularization)和[一篇博客](http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/)挺有帮助的。

### Dropout
随机扔掉传向末尾FC层的信号，使得末尾FC层不能完全相信所得输入。这个方法神奇地提高了正确率。可以将其理解为一种 Week Learner Ensemble 的方法。

想深究的同学[插这里](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

### Update Function / Optimization Function
1. 普通的 Gradient Descent
2. Momentum Update
3. Adam Update

还有其他的 Update。但是这三个足以把一些原则性的问题讲清楚，所以教程就选择了这三个。
