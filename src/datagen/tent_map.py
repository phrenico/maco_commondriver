import numpy as np

def tents(x, A):
  B = np.linalg.pinv((np.linalg.pinv(A)-np.eye(len(x))))
  c = - B.sum(axis=1)
  x_b = np.linalg.pinv(A).sum(axis=1)
  # print('xb:', x_b)
  is_smaller = ((x - x_b) < 0).astype(int)
  # print(is_smaller)
  xn = is_smaller * np.dot(A, x) + (1-is_smaller) * (np.dot(B, x) + c)
  # print(x, xn)
  return vec_reflect(xn)

gen_tentmap =