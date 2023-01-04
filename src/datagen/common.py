import numpy as np

def vec_reflect(x, lb=0, ub=1):
  is_bigger = (x > lb)
  is_negative = (x < ub)
  is_normal = np.logical_and(x>lb, x<ub)
  return is_normal * x + is_bigger * (1 - (x%1)) + is_negative * np.abs(x)