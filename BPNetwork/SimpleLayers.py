# coding: utf-8

# 简单层的实现

import numpy as np

# 乘法层
class MulLayer:
  
  # 双输入
  def __init__(self):
    self.x = None
    self.y = None
  
  # 前向传播
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = x * y
    return out
  
  # 后向传播
  def backward(self, dout):
    # dout是上游传回来的导数值
    dx = dout * self.y # 偏导xy对x = y
    dy = dout * self.x
    return dx, dy
  
# 加法层
class AddLayer:
  
  # 双输入
  def __init__(self, x, y):
    self.x = x
    self.y = y
  
  def forward(self, x, y):
    self.x = x
    self.y = y
    out = self.x + self.y
    return out

  def backward(self, dout):
    dx = dout * 1 # 偏导x+y对x = 1
    dy = dout * 1
    return dx, dy

# Affine仿射变换层
class Affine:
  
  def __init__(self, W, b):
    self.W = W
    self.b = b
    self.x = None
    self.dW = None
    self.db = None

  def forward(self, x):
    self.x = x
    out = np.dot(x, self.W) + self.b
    return out
  
  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)
    return dx
