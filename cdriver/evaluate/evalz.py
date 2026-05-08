import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

def eval_lin(X, Y):
    regmod = LinearRegression()
    regmod.fit(X, Y)
    regmod2 = LinearRegression()
    regmod2.fit(Y, X)
    print(regmod.score(X, Y), regmod2.score(Y, X))


    return regmod, regmod2

def comp_ccorr(y, y_recon):
  """Computes crosscorrelation
  """
  T = len(y)
  c = np.correlate(1/T * scale(y[:T]), scale(y_recon), mode='full')
  tau = np.arange(-T+1, T)
  return tau, c

def get_maxes(tau, c):
  """Gets the argmax and max of |xcorr|

  param list[int] tau: lags
  param list[float] c: crosscorrelation values
  returns: lag, max
  rtype: int, float
  """
  i = np.argmax(np.abs(c))
  return tau[i], np.abs(c)[i]
