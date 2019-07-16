# coding: utf-8

# 一系列损失函数的实现

import numpy as np

# 均方误差
def MSELoss(y, t):
  return 0.5 * np.sum((y - t) ** 2)

# 交叉熵误差(支持Mini-Batch)
def CrossEntropyLoss(y, t, delta=1e-7, one_hot=True):
  if y.ndim == 1: # 单个数据
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
  batch_size = y.shape[0]
  if one_hot:
    # y内是独热向量(one-hot)
    # 引入delta防止np.log(0)
    return -np.sum(t * np.log(y + delta)) / batch_size
    # np.log是ln
  else:
    # y是Label本身
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta))