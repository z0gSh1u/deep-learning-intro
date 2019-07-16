# coding: utf-8

# 激活函数层的实现

import numpy as np
import sys, os
sys.path.append('f:\\KerasLearning\\PythonImpl\\')

# ReLU层
class ReLU:
  
  def __init__(self):
    self.mask = None
  
  def forward(self, x):
    self.mask = (x <= 0)
    out = x.copy()
    out[self.mask] = 0
    return out
  
  def backward(self, dout):
    dout[self.mask] = 0
    dx = dout
    return dx

# Sigmoid层
class Sigmoid:

  def __init__(self):
    self.out = None
  
  def forward(self, x):
    out = 1 / (1 + np.exp(-x))
    self.out = out
    return out
  
  def backward(self, dout):
    dx = dout * (1.0 - self.out) * self.out
    return dx

# Softmax-with-Loss层
from NeuralNetwork import ActivationFunction
from Learning import LossFunction
class SoftmaxWithLoss:
  
  def __init__(self):
    self.loss = None
    self.y = None
    self.t = None
  
  def forward(self, x, t):
    self.t = t
    self.y = ActivationFunction.Softmax(x)
    self.loss = LossFunction.CrossEntropyLoss(self.y, self.t)
    return self.loss
  
  def backward(self, dout=1):
    batch_size = self.t.shape[0]
    dx = (self.y - self.t) / batch_size
    return dx
