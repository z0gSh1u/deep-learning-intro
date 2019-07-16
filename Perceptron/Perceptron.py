# coding: utf-8

# 本例程使用简单的感知机来完成一些门电路

import numpy as np

def Perceptron(w1, w2, b, x1, x2):
  x = np.array([x1, x2])
  w = np.array([w1, w2])
  y = np.sum(w * x) + b
  if y <= 0:
    return 0
  else:
    return 1

def AND(x1, x2):
  return Perceptron(
    0.5, 0.5,
    -0.7,
    x1, x2
  )

def NAND(x1, x2):
  return Perceptron(
    -0.5, -0.5,
    0.7,
    x1, x2
  )
  # 或者简单地调用 NOT AND(x1, x2)

def OR(x1, x2):
  return Perceptron(
    0.5, 0.5,
    -0.2,
    x1, x2
  )

'''
  可以简单地认为，神经网络就是在自动地寻找合适的w1, w2, b
'''

def XOR(x1, x2):
  Layer_0 = [x1, x2] # 所谓的 输入层
  Layer_1 = [NAND(Layer_0[0], Layer_0[1]), 
             OR(Layer_0[0], Layer_0[1])] # 所谓的 隐藏层
  Layer_2 = [AND(Layer_1[0], Layer_1[1])] # 所谓的 输出层
  return Layer_2[0]

def test():
  print("XOR(0, 0) = ", XOR(0, 0))
  print("XOR(0, 1) = ", XOR(0, 1))
  print("XOR(1, 0) = ", XOR(1, 0))
  print("XOR(1, 1) = ", XOR(1, 1))

if __name__ == '__main__':
  test()