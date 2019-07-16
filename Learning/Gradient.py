# coding: utf-8

# 数值梯度计算

import numpy as np

def numerical_gradient(f, x, delta=1e-4):
  if delta < 1e-10:
    print("[numerical_gradient WARN]: `delta` might be to small, causing round-off error.")
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    # 计算f(x+h)
    x[idx] = tmp_val + delta
    fxh1 = f(x)
    # 计算f(x-h)
    x[idx] = tmp_val - delta
    fxh2 = f(x)
    # 计算梯度
    grad[idx] = (fxh1 - fxh2) / (2 * delta)
    x[idx] = tmp_val # 还原
    it.iternext()
  return grad

def gradient_descent(f, init_x, step_num, lr=0.01, delta=1e-4):
  x = init_x
  for i in range(step_num):
    grad = numerical_gradient(f, x, delta)
    x -= lr * grad
  return x

def test():
  def func(x):
    # f(x1,x2)=x0^2+x1^2
    return x[0]**2 + x[1]**2
  init_x = np.array([-3., 4.])
  print(gradient_descent(
    func, init_x, step_num=1000))

if __name__ == "__main__":
  test()