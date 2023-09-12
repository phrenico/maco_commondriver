import numpy as np
from pandas import DataFrame

class TentMap:
  def __init__(self, alpha, A, x0):
    self.a = alpha
    self.A = A
    self.x0 = x0

  def bounder(self, x):
    """Applies boundary condition on data"""
    z = np.array(x)
    while not np.all(np.logical_and(z<=1, z>=0)):
        z = np.abs(z)
        inds  = np.where(z > 1)[0]
        for i in inds:
            z[i] = 1 - (z[i] - 1)
    return z

  def step(self, x):
    alpha = self.a
    A = self.A

    y = A @ x
    b = y <= (1/alpha)
    z =  b * alpha * y + (1 - b) * (alpha / (alpha -1) ) * (1 - y)
    return self.bounder(z)

  def gen_dataset(self, n):
    x = [self.x0]
    for i in range(n-1):
      x.append(self.step(x[-1]))
    return np.array(x)

class TentMapExpRunner:
  def __init__(self, nvars, baseA, a_interval):
    self.nvars = nvars
    self.baseA = baseA
    self.a_interval = a_interval

  def sample_params(self, A=None, a=None, x0=None, min_c_strenght=0.1, seed=None):
    np.random.seed(seed)
    aint = self.a_interval
    if a is None:
      a = aint[0] + (aint[1]-aint[0]) * np.random.rand(self.nvars)

    if A is None:
      A = self.baseA * ( (1 - min_c_strenght) * np.random.rand(self.nvars**2).reshape([self.nvars, self.nvars]) + min_c_strenght )
    np.fill_diagonal(A, 1)

    if x0 is None:
      x0 = np.random.rand(self.nvars)
    self.A = A
    self.a = a
    self.x0 = x0
    return a, A, x0

  def gen_experiment(self, n, A=None, a=None, x0=None, min_c_strength=0.1, seed=None):
    a, A, x0 = self.sample_params(a=a, A=A, x0=x0, min_c_strenght=min_c_strength, seed=seed)
    data = TentMap(alpha=a, A=A, x0=x0).gen_dataset(n)
    self.data = data
    return data, {'a': a, 'A': A, 'x0': x0}

if __name__ == "__main__":
  # Generate time series
  n = 20_000
  m = 3

  bA = np.eye(m)
  bA[1, 0] = 1
  bA[2, 0] = 1

  aint = [2, 10]

  x, d = TentMapExpRunner(m, baseA=bA, a_interval=aint).gen_experiment(n, seed=0)

  # Save out results
  save_fname = '../../data/tent_1d_data.csv'
  x_df = DataFrame(x)
  x_df.to_csv(save_fname)
