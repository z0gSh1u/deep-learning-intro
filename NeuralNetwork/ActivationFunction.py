# coding: utf-8

# 一系列激活函数的实现

import numpy as np

# 阶跃函数(阈值TH = 0)
def step_func(x):
  y = (x > 0)
  return y.astype(np.int)

# Sigmoid函数
def Sigmoid(x):
  return 1 / (1 + np.exp(-x))

# ReLU函数
def ReLU(x):
  return np.maximum(0, x)

# Softmax函数
def Softmax(x):
  # 取对数防止溢出问题
  '''
    在进行Softmax的指数函数的运算时，加上（或者减去）某个常数并不会改变运算的结果。
    这里的C可以使用任何值，但是为了防止溢出，一般会使用输入信号中的最大值去减
  '''
  c = np.max(x)
  exp_x = np.exp(x - c)
  sum_exp_x = np.sum(exp_x)
  return exp_x / sum_exp_x

