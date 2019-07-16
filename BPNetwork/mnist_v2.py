# coding: utf-8
# TODO: NEED DEBUG

# 使用层次化的BP神经网络进行MNIST手写数字识别

import sys
# sys.path.append('../') # 从而可以import上级目录的包
sys.path.append('f:\\KerasLearning\\PythonImpl\\')
import numpy as np
from BPNetwork.Layers import *
from collections import OrderedDict

# BP神经网络
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
    self.params['W2'] = W_init_std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size) + b_init_std
    # 开始构造层
    self.layers = OrderedDict()
    self.layers['Affine1'] = \
      SimpleLayers.Affine(self.params['W1'], self.params['b1'])
    self.layers['ReLU1'] = \
      ActivationFunctionLayers.ReLU()
    self.layers['Affine2'] = \
      SimpleLayers.Affine(self.params['W2'], self.params['b2'])
    self.lastLayer = \
      ActivationFunctionLayers.SoftmaxWithLoss()

  def predict(self, x):
    for layer in self.layers.values():
      x = layer.forward(x)
    return x
  
  def loss(self, x, t):
    y = self.predict(x)
    return self.lastLayer.forward(y, t)
  
  def accuracy(self, x, t):
    y = self.predict(x)
    y = np.argmax(y, axis=1)
    t = np.argmax(t, axis=1)
    accuracy = np.sum(y == t) / float(x.shape[0])
    return accuracy
  
  # 利用反向传播的结论计算梯度，而非数值梯度
  def grad(self, x, t):
    self.loss(x, t)
    dout = 1
    dout = self.lastLayer.backward(dout)
    layers = list(self.layers.values())
    layers.reverse()
    for layer in layers:
      dout = layer.backward(dout)
    # 组织输出
    grads = {}
    grads['W1'] = self.layers['Affine1'].dW
    grads['b1'] = self.layers['Affine1'].db
    grads['W2'] = self.layers['Affine2'].dW
    grads['b2'] = self.layers['Affine2'].db
    return grads


# 应用MNIST数据集测试
from Dataset.MNIST.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = \
  load_mnist(normalize=True, one_hot_label=True)

net = Net(
  input_size=784,
  hidden_size=50,
  output_size=10
)

ITERS_NUM = 10000
TRAIN_SIZE = x_train.shape[0]
BATCH_SIZE = 100
LR = 0.1

print('[Start training...]')
rounds = 0
for i in range(ITERS_NUM):
  batch_subs = np.random.choice(TRAIN_SIZE, BATCH_SIZE)
  x_batch = x_train[batch_subs]
  t_batch = t_train[batch_subs]

  # 计算梯度，更新参数
  grad = net.grad(x_batch, t_batch)
  for key in ('W1', 'b1', 'W2', 'b2'):
    net.params[key] -= LR * grad[key]
  
  # 每500次循环打印一次效果
  if i % 500 == 0:
    rounds += 1
    loss = net.loss(x_batch, t_batch)
    accu_on_test = net.accuracy(x_test, t_test)
    print('[ROUND %d] LOSS=%f, ACCU=%f' % (rounds, loss, accu_on_test))
