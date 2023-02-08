import numpy as np
from .common import gen_ts, vec_reflect
from pandas import DataFrame

def tents(x, A):
  """Update rule for the tent map (possibli coupled tent-maps)

  :param numpy.ndarray x: current state
  :param numpy.ndarray A: tentmap and coupling matrix
  :return: next values
  """
  B = np.linalg.pinv((np.linalg.pinv(A)-np.eye(len(x))))
  c = - B.sum(axis=1)
  x_b = np.linalg.pinv(A).sum(axis=1)
  # print('xb:', x_b)
  is_smaller = ((x - x_b) < 0).astype(int)
  # print(is_smaller)
  xn = is_smaller * np.dot(A, x) + (1-is_smaller) * (np.dot(B, x) + c)
  # print(x, xn)
  return vec_reflect(xn)

gen_tentmap = gen_ts(tents)

if __name__ == "__main__":
  # Generate time series
  m = 3
  A = [2.3, 3, 4] * np.eye(m)
  A[1, 0] = 0.8
  A[2, 0] = 2
  print(A)
  x0 = np.random.rand(m)
  n = 20000

  x = gen_tentmap(x0, n, f_kwargs=dict(A=A))

  # Save out results
  save_fname = '../../data/tent_1d_data.csv'
  x_df = DataFrame(x)
  x_df.to_csv(save_fname)
