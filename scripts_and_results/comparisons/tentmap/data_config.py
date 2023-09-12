'''
Configuration file for generating the tent map datasets
'''
import numpy as np

N = 50
n = 20_000

aint = (2, 10.)  # interval to chose from the value of r parameter
A0 = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])  # basic connection structure