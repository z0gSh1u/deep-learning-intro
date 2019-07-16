# coding: utf-8

# 使用最简单的神经网络实现MNIST手写数字识别

import sys
# sys.path.append('../') # 从而可以import上级目录的包
sys.path.append('f:\\KerasLearning\\PythonImpl\\')

import numpy as np
from NeuralNetwork import ActivationFunction
from Learning import LossFunction, Gradient

# 仅一个隐藏层的仅前向传播的简单神经网络
class Net:

  # 初始化网络骨架
  def __init__(self, 
  input_size, # 输入层神经元数
  hidden_size, # 隐藏层神经元数
  output_size, # 输出层神经元数
  W_init_std=0.01, # 权重初始化调节系数
  b_init_std=0 # 偏置初始化扰动值
  ):
    self.params = {} # 神经网络参数
    # W: 权重 b: 偏置 act: 激活函数
    self.params['W1'] = W_init_std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size) + b_init_std
    self.params['act1'] = ActivationFunction.Sigmoid
    self.params['W2'] = W_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size) + b_init_std
    self.params['act2'] = ActivationFunction.Softmax

  # 进行一轮预测
  def predict(self, x):
    W1, W2 = self.params['W1'], self.params['W2']
    b1, b2 = self.params['b1'], self.params['b2']
    a1 = np.dot(x, W1) + b1
    z1 = self.params['act1'](a1)
    a2 = np.dot(z1, W2) + b2
    y = self.params['act2'](a2)
    return y

  # 定义损失函数
  def loss(self, x, t):
    y = self.predict(x)
    # 注意喂入one_hot化的MNIST标签数据
    return LossFunction.CrossEntropyLoss(y, t, one_hot=True)
  
  # 计算预测准确度
  def accuracy(self, x, t):
    y = self.predict(x) # y=[0.1, 0.7, 0.15 ...]对应为0~9预测把握
    y = np.argmax(y, axis=1) # 取出最大值索引，即为预测值
    t = np.argmax(t, axis=1) # t是独热向量，最大值索引即为正确值
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy

  # 对于整个网络的梯度计算
  def grad(self, x, t):
    # 请结合lambda函数理解该行(loss_W是一个函数而非一个值)
    loss_W = lambda W: self.loss(x, t)
    grads = {}
    grads['W1'] = Gradient.numerical_gradient(loss_W, self.params['W1'])
    grads['b1'] = Gradient.numerical_gradient(loss_W, self.params['b1'])
    grads['W2'] = Gradient.numerical_gradient(loss_W, self.params['W2'])
    grads['b2'] = Gradient.numerical_gradient(loss_W, self.params['b2'])
    return grads


# 导入MNIST数据集
from Dataset.MNIST.mnist import load_mnist
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# 设定超参数
ITERS_SUM = 200
TRAIN_SIZE = x_train.shape[0]
BATCH_SIZE = 100
LR = 0.1
# 开始训练
loss_recorder = []
net = Net(
  input_size=784, # 28*28
  hidden_size=10, # 自定义
  output_size=10 # 10个数
)

rounds = 0
print("[Start training...]")
for i in range(ITERS_SUM):
  # Mini-Batch
  # 注意到choice是随机的，所以我们不需要shuffle数据集
  # 但这也导致了可能存在个别数据自始至终不会被喂入网络
  batch_subs = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
  x_batch = x_train[batch_subs]
  t_batch = t_train[batch_subs]
  # 网络参数更新
  grad = net.grad(x_batch, t_batch)
  for key in ('W1', 'b1', 'W2', 'b2'):
    net.params[key] -= LR * grad[key]  
  # 每20轮打印一次损失和精确度评估
  if i % 20 == 0:
    rounds += 1
    print("[Round %d]: Loss=%f, Accu=%f" % 
      (rounds, net.loss(x_batch, t_batch), net.accuracy(x_batch, t_batch))
    )
  # 利用该方法进行训练需要大量时间，作为演示，仅做200次循环
  '''
    [Round 1]: Loss=6.903439, Accu=0.150000
    [Round 2]: Loss=6.896936, Accu=0.160000
    [Round 3]: Loss=6.904256, Accu=0.050000
    [Round 4]: Loss=6.909626, Accu=0.080000
    [Round 5]: Loss=6.894128, Accu=0.140000
    [Round 6]: Loss=6.876732, Accu=0.200000
    [Round 7]: Loss=6.901391, Accu=0.190000
    [Round 8]: Loss=6.912426, Accu=0.100000
    [Round 9]: Loss=6.891001, Accu=0.150000
    [Round 10]: Loss=6.901920, Accu=0.110000

    [Done] exited with code=0 in 1137.267 seconds
  '''
  # 由于每轮batch的不同，所以虽然每轮都往梯度下降方向前进，
  # 但Loss和Accu不一定严格递减
  # 当参数合适的时候，经过大量训练，Accu的整体趋势是上升的
  