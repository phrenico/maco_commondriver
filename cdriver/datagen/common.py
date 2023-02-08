import numpy as np

def vec_reflect(x, lb=0, ub=1):
  """Reflects the values to the boundary (for small boundary violations)

  :param numpy.ndarray x: values
  :param float lb: lower boundary
  :param float ub: upper boundary
  :return: reflected values
  :rtype: numpy.ndarray
  """
  is_bigger = (x > ub)
  is_negative = (x < lb)
  is_normal = np.logical_and(x>lb, x<ub)
  return is_normal * x + is_bigger * (1 - (x%1)) + is_negative * np.abs(x)

def gen_ts(f):
  """decorator for generating time series from update rules (discrete-time dynamical systems)

  :param function f: update rule
  :return: function to generate time series from the update rule
  :rtype: function
  """
  def wrapper(x0, n, f_kwargs={}):
    """wrapper function to generate time series with an update rule

    :param numpy.ndarray x0: initial condition
    :param int n: number of datapoints with initial condition
    :param dict f_kwargs: keyword arguments for the update rule
    :return: time series
    :rtype: numpy.ndarray
    """
    x = [x0]
    for i in range(n-1):
      x.append(f(x[-1], **f_kwargs))
    return np.array(x)
  return wrapper