import numpy as np
from sklearn.linear_model import LinearRegression

def eval_lin(X, Y):
    regmod = LinearRegression()
    regmod.fit(X, Y)
    regmod2 = LinearRegression()
    regmod2.fit(Y, X)
    print(regmod.score(X, Y), regmod2.score(Y, X))


    return regmod, regmod2
